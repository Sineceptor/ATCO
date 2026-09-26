# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/build_master_vocab.py
# What it does: Counts words over all data files and sorts them into hand-written ATC categories to build decoding prompts.
# Known problems:
#   - test.jsonl is one of the input files and the CRITICAL_* lists were written from evaluation errors. Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import json
import os
import re
import collections
from tqdm import tqdm

# 配置路径
DATA_FILES = [
    "processed_data/train.jsonl",
    "processed_data/train_augmented.jsonl",
    "processed_data/train_synthetic_real_pro.jsonl",
    "processed_data/test.jsonl",
    # 如果有生成的纠错数据，也加进来
    "processed_data/train_synthetic_hard.jsonl",
]

OUTPUT_FILE = "processed_data/vocab_master.json"

# 预定义分类规则
# Classify by vocabulary rules; unmatched words become OTHER.

CATEGORIES = {
    "ICAO_ALPHABET": {
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
    },
    "NUMBERS": {
        "ZERO",
        "ONE",
        "TWO",
        "THREE",
        "FOUR",
        "FIVE",
        "SIX",
        "SEVEN",
        "EIGHT",
        "NINE",
        "TEN",
        "ELEVEN",
        "TWELVE",
        "THIRTEEN",
        "FOURTEEN",
        "FIFTEEN",
        "SIXTEEN",
        "SEVENTEEN",
        "EIGHTEEN",
        "NINETEEN",
        "TWENTY",
        "THIRTY",
        "FORTY",
        "FIFTY",
        "SIXTY",
        "SEVENTY",
        "EIGHTY",
        "NINETY",
        "HUNDRED",
        "THOUSAND",
        "DECIMAL",
        "POINT",
    },
    "COMMANDS_ACTIONS": {
        "CLEARED",
        "LAND",
        "TAKEOFF",
        "TAXI",
        "HOLD",
        "POSITION",
        "LINE",
        "UP",
        "CLIMB",
        "DESCEND",
        "MAINTAIN",
        "CONTACT",
        "REPORT",
        "TURN",
        "HEADING",
        "LEFT",
        "RIGHT",
        "DIRECT",
        "PROCEED",
        "APPROACH",
        "DEPARTURE",
        "SQUAWK",
        "IDENT",
        "PUSH",
        "START",
        "APPROVED",
        "CANCEL",
        "NEGAT",
        "AFFIRM",
        "ROGER",
        "WILCO",
        "COPY",
        "READBACK",
        "CORRECT",
        "CROSS",
        "VACATE",
        "BACKTRACK",
        "ESTABLISHED",
    },
    "ATC_ENTITIES": {
        "TOWER",
        "GROUND",
        "RADAR",
        "APPROACH",
        "CONTROL",
        "DELIVERY",
        "CENTER",
        "INFORMATION",
        "ATIS",
        "RUNWAY",
        "TWY",
        "TAXIWAY",
        "GATE",
        "STAND",
        "APRON",
        "QNH",
        "WIND",
        "KNOTS",
        "DEGREES",
        "VISUAL",
        "ILS",
        "LOC",
        "GLIDESLOPE",
        "MILES",
        "FEET",
        "FLIGHT",
        "LEVEL",
        "ALTITUDE",
    },
    # Locations and callsigns added from observed evaluation errors.
    "CRITICAL_LOCATIONS": {
        "STEFANIK",
        "LZIB",
        "BRATISLAVA",
        "SION",
        "LSGS",
        "RUZYNE",
        "LKPR",
        "PRAGUE",
        "BERN",
        "LSZB",
        "ZURICH",
        "LSZH",
        "BRNO",
        "LKTB",
        "TURANY",
        "KOSICE",
        "POPRAD",
        "OSTRAVA",
        "SYDNEY",
        "YSSY",  # 放在这里是为了监控它的频率
    },
    "CRITICAL_CALLSIGNS": {
        "MOSQUITO",
        "NAV",
        "CHECKER",
        "TATRA",
        "BAIR",
        "TIME",
        "AIR",
        "SLOVAK",
        "GOVERNMENT",
        "TRANS",
        "EUROPE",
        "SKY",
        "TRAVEL",
        "ELITE",
        "JET",
        "REGA",
        "HELI",
        "TWIN",
        "STAR",
        "LEGEND",
        "FRACTION",
        "SUNTURK",
        "AIR",
        "HAMBURG",
        "AIR",
        "NOSTRUM",
        "EUROTRANS",
        "WIZZ",
        "RYANAIR",
        "CSA",
        "LUFTHANSA",
        "SWISS",
        "AUSTRIAN",
        "EMIRATES",
        "QANTAS",
        "QATARI",
        "CHINA",
        "SOUTHERN",
        "EASTERN",
        "DYNASTY",
        "QLINK",
        "REX",
        "JETSTAR",
    },
    "GREETINGS_FOREIGN": {
        "GOOD",
        "DAY",
        "MORNING",
        "AFTERNOON",
        "EVENING",
        "BYE",
        "HELLO",
        "DOBRY",
        "DEN",
        "DOBRE",
        "RANO",
        "VECER",
        "AHOJ",
        "NA",
        "SLYSENOU",  # 捷克/斯洛伐克
        "BONJOUR",
        "BONNE",
        "SOIREE",
        "MERCI",
        "AUREVOIR",
        "TOUTE",  # 法语
        "GRUEZI",
        "SERVUS",
        "TSCHUSS",
        "BIS",
        "SPATER",
        "DANKE",
        "SCHONE",  # 德语
    },
}


def clean_word(w):
    # 去除标点，只留字母和数字
    return re.sub(r"[^A-Z0-9]", "", w.upper())


def classify_word(word):
    for cat, words in CATEGORIES.items():
        if word in words:
            return cat
    return "OTHER"  # 没在这个表里的


def main():
    print("开始构建全量词库...")

    vocab_counter = collections.Counter()

    # 扫描所有文件
    valid_files = [f for f in DATA_FILES if os.path.exists(f)]

    for file_path in valid_files:
        print(f"读取: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("ASR_transcript_clean", "")
                    if not text:
                        continue

                    # 拆分单词
                    words = text.split()
                    for w in words:
                        clean_w = clean_word(w)
                        if clean_w:
                            vocab_counter[clean_w] += 1
                except:
                    pass

    # 结构化输出
    structured_vocab = {cat: {} for cat in CATEGORIES.keys()}
    structured_vocab["OTHER"] = {}

    total_words = 0
    unique_words = 0

    for word, count in vocab_counter.most_common():
        cat = classify_word(word)
        structured_vocab[cat][word] = count
        total_words += count
        unique_words += 1

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_vocab, f, indent=4)

    # 打印分析报告
    print("\n" + "=" * 40)
    print("词库统计报告")
    print("=" * 40)
    print(f"总单词数 (Tokens): {total_words}")
    print(f"词汇量 (Unique):   {unique_words}")
    print("-" * 40)
    print(f"{'Category':<20} | {'Unique Words':<12} | {'Top 3 Examples'}")
    print("-" * 40)

    for cat, words_dict in structured_vocab.items():
        count = len(words_dict)
        # 取频率最高的前3个
        top3 = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join([f"{w}({c})" for w, c in top3])
        print(f'{cat:<20} | {count:<12} | {top3_str}')

    print("=" * 40)

    # 自动生成一个“词汇 Prompt”
    # 逻辑：提取 CRITICAL_LOCATIONS 和 CRITICAL_CALLSIGNS 里所有的词
    # Vocabulary prompt built from selected locations and callsigns.

    prompt_words = []
    prompt_words.extend(list(structured_vocab["CRITICAL_LOCATIONS"].keys()))
    prompt_words.extend(list(structured_vocab["CRITICAL_CALLSIGNS"].keys()))

    # 去重并打乱一点点，或者按频率排序
    prompt_str = ", ".join(prompt_words)

    print("\n【生成的词汇 Prompt (可直接用于推理)】")
    print("-" * 60)
    print(f"ATC Context. Valid vocabularies: {prompt_str}")
    print("-" * 60)

    print(f"\n详细词库已保存至: {OUTPUT_FILE}")
    print("Check the OTHER category for unclassified callsigns.")


if __name__ == "__main__":
    main()
