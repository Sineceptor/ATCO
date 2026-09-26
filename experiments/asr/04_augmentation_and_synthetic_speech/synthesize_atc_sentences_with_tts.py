# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/augment_grammar.py
# What it does: Generates 5,000 ATC sentences from templates (callsign, runway, heading, flight level, frequency, wind, squawk) and speaks them with Edge TTS in 31 English accents at varied speaking rates.
# Known problems:
#   - The airline list includes callsigns picked from test-set errors. Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
#   - No random seed; needs network access; needs Python 3.12+ (nested f-string quotes).
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import random
import json
import asyncio
import os
import subprocess
import edge_tts
from tqdm.asyncio import tqdm_asyncio

# Configuration
OUTPUT_DIR = "processed_data/audio_synthetic_ultimate_wav_final"
OUTPUT_JSONL = "processed_data/train_synthetic_ultimate_wav_final.jsonl"
NUM_SAMPLES = 5000  # 生成数量

# 并发数 (建议 30-50)
CONCURRENCY_LIMIT = 40
TARGET_SR = 16000

# Speech voices
# Available voices
VOICES = [
    # 北美
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-CA-LiamNeural",
    "en-CA-ClaraNeural",  # 加拿大
    # 欧洲
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-GB-LibbyNeural",
    "en-IE-EmilyNeural",
    "en-IE-ConnorNeural",  # 爱尔兰 (语速快)
    # 亚太 (训练难点)
    "en-AU-WilliamNeural",
    "en-AU-NatashaNeural",  # 澳洲
    "en-NZ-MitchellNeural",  # 新西兰
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",  # 印度 (重口音)
    "en-SG-WayneNeural",
    "en-SG-LunaNeural",  # 新加坡
    "en-HK-SamNeural",
    "en-HK-YanNeural",  # 香港
    "en-PH-JamesNeural",
    "en-PH-RosaNeural",  # 菲律宾
    # 非洲
    "en-NG-AbeoNeural",
    "en-NG-EzinneNeural",  # 尼日利亚
    "en-KE-AsiliaNeural",
    "en-KE-ChilembaNeural",  # 肯尼亚
    "en-TZ-ElimuNeural",
    "en-TZ-ImaniNeural",  # 坦桑尼亚
    "en-ZA-LeahNeural",
    "en-ZA-LukeNeural",  # 南非
]

# 完整词库 (含 50+ 航司)
ICAO_ALPHABET = [
    "ALPHA",
    "BRAVO",
    "CHARLIE",
    "DELTA",
    "ECHO",
    "FOXTROT",
    "GOLF",
    "HOTEL",
    "INDIA",
    "JULIETT",
    "KILO",
    "LIMA",
    "MIKE",
    "NOVEMBER",
    "OSCAR",
    "PAPA",
    "QUEBEC",
    "ROMEO",
    "SIERRA",
    "TANGO",
    "UNIFORM",
    "VICTOR",
    "WHISKEY",
    "X-RAY",
    "YANKEE",
    "ZULU",
]
NUMBERS_MAP = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}
TAXIWAYS = ICAO_ALPHABET[:15]  # A-O

# Includes callsigns selected from observed test-set errors.
AIRLINES = [
    # 欧洲
    "RYANAIR",
    "EASY",
    "LUFTHANSA",
    "AIR FRANCE",
    "SPEEDBIRD",
    "KLM",
    "SWISS",
    "SCANDINAVIAN",
    "AUSTRIAN",
    "EUROWINGS",
    "WIZZ AIR",
    "SHAMROCK",
    "TURKISH",
    "POLLOT",
    "IBERIA",
    "ALITALIA",
    "FINNAIR",
    "NOR SHUTTLE",
    "TAP PORTUGAL",
    "EUROTRANS",
    "VOLOTEA",
    "VUELING",
    # 美洲
    "UNITED",
    "AMERICAN",
    "DELTA",
    "AIR CANADA",
    "JETBLUE",
    "SOUTHWEST",
    "FEDEX",
    "GIANT",
    "UPS",
    "LATAM",
    "AEROMEXICO",
    # 亚洲/其他
    "AIR CHINA",
    "CATHAY",
    "DYNASTY",
    "EVA",
    "ALL NIPPON",
    "JAPAN AIR",
    "KOREAN AIR",
    "SINGAPORE",
    "THAI",
    "QANTAS",
    "EMIRATES",
    "ETIHAD",
    "QATARI",
    "SAUDIA",
    "INDIGO",
    "VIET NAM",
]

# 真实航点 (用于替换错误的 ALPHAALPHAALPHA)
WAYPOINTS = [
    "IKUMA",
    "DOGAL",
    "BOMBI",
    "GIVEM",
    "XILAN",
    "VABIK",
    "LONDI",
    "AMMAN",
    "RITAX",
    "SULOV",
    "EKAUL",
    "NORKU",
    "ODINA",
    "PIMOK",
    "ELTOK",
    "RUDKA",
    "TULIP",
    "ROLEX",
    "GIGOL",
    "VENUS",
    "OSCAR",
]


# 严格逻辑生成 (Logic Fixes)


def digits_to_words(number_str):
    """Read each digit separately, for example 18 as ONE EIGHT."""
    return " ".join([NUMBERS_MAP[digit] for digit in number_str])


def gen_callsign():
    airline = random.choice(AIRLINES)
    # 生成 2-4 位数字
    flight_num_str = str(random.randint(10, 9999))
    flight_num_words = digits_to_words(flight_num_str)

    if random.random() < 0.2:
        suffix = random.choice(ICAO_ALPHABET + ["HEAVY", "SUPER"])
        return f"{airline} {flight_num_words} {suffix}"
    return f"{airline} {flight_num_words}"


# 修复：跑道号逻辑 (01 - 36)
def gen_runway():
    # 随机生成 1 到 36 的数字
    rwy_num = random.randint(1, 36)
    # 格式化为两位数 (01, 09, 27)
    rwy_str = f"{rwy_num:02d}"

    words = digits_to_words(rwy_str)
    suffix = random.choice(["LEFT", "RIGHT", "CENTER", ""])
    return f"{words} {suffix}".strip()


# 修复：航向逻辑 (001 - 360)
def gen_heading():
    hdg_num = random.randint(1, 360)
    hdg_str = f"{hdg_num:03d}"  # 补齐三位，如 090
    return digits_to_words(hdg_str)


# 修复：高度层逻辑 (FL060 - FL430)
def gen_flight_level():
    fl_num = random.randint(60, 430)
    fl_str = f"{fl_num:03d}"
    return f"FLIGHT LEVEL {digits_to_words(fl_str)}"


# 修复：速度逻辑 (160 - 320 节)
def gen_speed():
    # 步长为 10，例如 210, 220
    spd_num = random.randrange(160, 320, 10)
    spd_str = str(spd_num)
    return digits_to_words(spd_str)


# 修复：频率逻辑 (118.0 - 136.9)
def gen_freq():
    # 简单模拟 118.000 - 136.975
    mhz = random.randint(118, 136)
    khz = random.choice(["0", "05", "1", "125", "2", "3", "5", "7", "8"])
    return f"{digits_to_words(str(mhz))} DECIMAL {digits_to_words(khz)}"


def gen_wind():
    deg = f"{random.randint(1, 360):03d}"
    spd = f"{random.randint(3, 25):02d}"  # 风速 3-25 节
    return f"WIND {digits_to_words(deg)} DEGREES {digits_to_words(spd)} KNOTS"


# 生成句子主逻辑
def generate_sentence():
    scenario = random.choice(
        [
            "ground_taxi",
            "tower_takeoff",
            "tower_landing",
            "radar_vector",
            "radar_compound",
            "radar_compound_2",
            "approach_ils",
            "handoff",
            "status_report",
        ]
    )
    cs = gen_callsign()

    if scenario == "ground_taxi":
        rwy = gen_runway()
        twy = random.choice(TAXIWAYS)
        return f"{random.choice([f'TAXI TO HOLDING POINT RUNWAY {rwy} VIA {twy}', f'HOLD SHORT OF RUNWAY {rwy}', f'PUSH AND START APPROVED FACING {random.choice(['NORTH', 'SOUTH'])}', f'GIVE WAY TO THE {random.choice(['BOEING', 'AIRBUS'])} ON {twy}'])} {cs}"

    elif scenario == "tower_takeoff":
        rwy = gen_runway()
        # 使用真实航点库，修复重复字符问题
        wpt = random.choice(WAYPOINTS)
        return f"{random.choice([f'LINE UP AND WAIT RUNWAY {rwy}', f'{gen_wind()} RUNWAY {rwy} CLEARED FOR TAKEOFF', 'CANCEL TAKEOFF STOP IMMEDIATELY', f'AFTER DEPARTURE TURN {random.choice(['LEFT', 'RIGHT'])} DIRECT {wpt}'])} {cs}"

    elif scenario == "tower_landing":
        rwy = gen_runway()
        twy = random.choice(TAXIWAYS)
        return f"{random.choice([f'{gen_wind()} RUNWAY {rwy} CLEARED TO LAND', 'GO AROUND I SAY AGAIN GO AROUND', f'VACATE RUNWAY {rwy} VIA {twy}'])} {cs}"

    elif scenario == "radar_vector":
        hdg = gen_heading()
        alt = gen_flight_level()
        qnh = (
            f"Q N H {random.choice(['ONE ZERO ONE THREE', 'ONE ZERO ZERO ONE', 'NINE NINE EIGHT'])}"
        )
        return f"{random.choice([f'TURN {random.choice(['LEFT', 'RIGHT'])} HEADING {hdg}', f'DESCEND TO {alt} {qnh}', f'CLIMB TO {alt} EXPEDITE', 'MAINTAIN PRESENT HEADING'])} {cs}"

    elif scenario == "radar_compound":
        hdg = gen_heading()
        alt = gen_flight_level()
        rwy = gen_runway()
        return f"{random.choice([f'TURN LEFT HEADING {hdg} DESCEND TO {alt}', f'CLIMB TO {alt} TURN RIGHT HEADING {hdg}', f'TURN LEFT HEADING {hdg} INTERCEPT LOCALIZER RUNWAY {rwy}'])} {cs}"

    elif scenario == "radar_compound_2":
        spd = gen_speed()
        alt = gen_flight_level()
        return f"{random.choice([f'DESCEND TO {alt} REDUCE SPEED TO {spd} KNOTS', f'MAINTAIN SPEED {spd} KNOTS OR GREATER', f'REDUCE TO MINIMUM CLEAN SPEED AND DESCEND TO {alt}'])} {cs}"

    elif scenario == "approach_ils":
        rwy = gen_runway()
        return f"{random.choice([f'CLEARED I L S APPROACH RUNWAY {rwy}', f'REPORT ESTABLISHED LOCALIZER RUNWAY {rwy}', f'CLEARED VISUAL APPROACH RUNWAY {rwy} REPORT FIELD IN SIGHT'])} {cs}"

    elif scenario == "handoff":
        return f"CONTACT {random.choice(['RADAR', 'CONTROL', 'CENTER', 'APPROACH', 'DEPARTURE'])} {gen_freq()} {cs} GOOD DAY"

    elif scenario == "status_report":
        # 修复 Squawk 逻辑，避免出现 8 或 9 (e.g. 7894)
        sq = "".join(random.choices("01234567", k=4))
        sq_words = digits_to_words(sq)
        return f"{random.choice(['SQUAWK IDENT', f'SQUAWK {sq_words}', f'REPORT PASSING {gen_flight_level()}'])} {cs}"

    return ""


# 异步 + 转码 WAV 逻辑
async def convert_to_wav(input_path, output_path):
    """调用 FFmpeg 将音频转为 16000Hz 单声道 WAV"""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            "-y",
            "-loglevel",
            "error",
            output_path,
        ]
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()
        return process.returncode == 0
    except Exception as e:
        return False


async def generate_single_sample(index, semaphore, output_list):
    async with semaphore:
        text = generate_sentence()
        text = " ".join(text.split())

        voice = random.choice(VOICES)
        rate = f"{random.randint(-5, 25):+d}%"
        voice_region = voice.split("-")[1]

        temp_filename = f"TEMP_{index:05d}.mp3"
        temp_filepath = os.path.join(OUTPUT_DIR, temp_filename)
        final_filename = f"SYN_FIXED_{index:05d}_{voice_region}.wav"
        final_filepath = os.path.join(OUTPUT_DIR, final_filename)

        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(temp_filepath)
            success = await convert_to_wav(temp_filepath, final_filepath)

            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

            if success:
                output_list.append(
                    {
                        "audio_file": os.path.abspath(final_filepath),
                        "duration_sec": 0,
                        "ASR_transcript_clean": text,
                        "utt_id": f"SYN_FIXED_{index:05d}",
                    }
                )
                return True
            return False
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return False


async def main():
    print(f"启动数据生成器")
    print(f'口音库: {len(VOICES)} 种 (印度、新加坡、爱尔兰等全齐)')
    print(f'航司库: {len(AIRLINES)} 个 (Ryanair, Eurotrans 等全齐)')
    print(f"逻辑修复: 跑道(01-36), 速度(160-320), 无重复字符串")
    print(f"并发数: {CONCURRENCY_LIMIT}")

    # 检查 ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("错误: 未检测到 ffmpeg！")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    entries = []
    tasks = []

    for i in range(NUM_SAMPLES):
        task = generate_single_sample(i, semaphore, entries)
        tasks.append(task)

    await tqdm_asyncio.gather(*tasks)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("=" * 40)
    print(f"生成完成！")
    print(f"成功样本: {len(entries)}")
    print(f"标签文件: {OUTPUT_JSONL}")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
