# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/prepare_ner_data_strict.py
# What it does: First rule labeller: airline list, NATO alphabet, command verbs, number words, session callsigns/waypoints -> BIO tags. Version 2 (with stopwords) is training/entity_labels.py.
# Known problems:
#   - Any alphabetic word of 3+ letters that is in no list becomes WAYPOINT (the isupper() test is applied to an already-uppercased word), so 'hello' and 'and' are waypoints.
#   - Writes to the same folder as version 2, so which rules produced each saved model is uncertain.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import glob
import random
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed_data", "merged_sessions")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data", "ner_dataset_ultimate")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Vocabulary used by the labelling rules.


# 航空公司与常用呼号 (扩充版)
AIRLINES = {
    "jetstar",
    "qantas",
    "velocity",
    "virgin",
    "rex",
    "united",
    "singapore",
    "emirates",
    "cathay",
    "air",
    "china",
    "delta",
    "american",
    "shamrock",
    "speedbird",
    "lufthansa",
    "korean",
    "japan",
    "ana",
    "fedex",
    "ups",
    "british",
    "airways",
    "klm",
    "jump",
    "run",
    "medevac",
    "rescue",
    "polair",
    "chopper",
    "cessna",
    "piper",
    "tiger",
    "fiji",
    "tasman",
    "link",
    "nz",
    "new",
    "zealand",
    "bonza",
}

# ICAO 北约音标 (识别 Hotel Golf Papa 这种呼号的关键)
ICAO_PHONETICS = {
    "alfa",
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "xray",
    "yankee",
    "zulu",
}

# 动作指令 (Commands)
ATC_COMMANDS = {
    "contact",
    "climb",
    "descend",
    "turn",
    "maintain",
    "cleared",
    "land",
    "takeoff",
    "hold",
    "report",
    "squawk",
    "approach",
    "direct",
    "heading",
    "cross",
    "track",
    "identified",
    "cancel",
    "verify",
    "correction",
    "wind",
    "runway",
    "taxi",
    "pushback",
    "line",
    "up",
    "wait",
    "go",
    "around",
    "continue",
    "expect",
    "speed",
    "reduce",
    "increase",
    "intercept",
    "establish",
    "pass",
    "enter",
    "vacate",
    "monitor",
    "approve",
    "approved",
    "check",
    "confirm",
    "negative",
    "affirm",
    "leaving",
    "reaching",
}

# 数值与单位
NUMBERS_AND_UNITS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "zero",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "thousand",
    "hundred",
    "decimal",
    "point",
    "degrees",
    "knots",
    "feet",
}

# 特殊复合词处理 (这些词在一起时，应该被视为一个整体)
# 比如 "flight level" 应该被视为数值的前缀，或者 "left" "right" 是数值的后缀
VALUE_MODIFIERS = {"flight", "level", "left", "right", "center", "centre"}


def is_number_or_modifier(w):
    return w.lower() in NUMBERS_AND_UNITS or w.isdigit() or w.lower() in VALUE_MODIFIERS


def is_nato(w):
    return w.lower() in ICAO_PHONETICS


def tag_entities(text, local_context):
    """Assign BIO labels using the existing vocabulary and context rules."""
    clean_text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    words = clean_text.split()
    labels = ["O"] * len(words)

    # 提取 Context
    local_callsigns = set()
    if "callsigns" in local_context:
        for cs in local_context["callsigns"]:
            if isinstance(cs, str):
                local_callsigns.add(cs.split()[0].lower())
            elif isinstance(cs, dict):
                local_callsigns.add(cs["spoken"].split()[0].lower())

    local_waypoints = set(local_context.get("waypoints", []))

    i = 0
    while i < len(words):
        w_lower = words[i].lower()
        w_upper = words[i].upper()

        # 逻辑 1: 呼号 (Airline 或 NATO 序列)
        # 判定条件：是航司名，或者是 Context 里的呼号，或者是 NATO 音标且后面还有音标/数字
        is_callsign_start = (w_lower in AIRLINES) or (w_lower in local_callsigns)

        # 特殊处理：如果是一个孤立的 NATO 词 (如 "Mike")，可能是航路点；
        # This heuristic treats consecutive phonetic words as a callsign.
        if not is_callsign_start and is_nato(w_lower):
            # 只有当它是连续 NATO 序列的一部分，或者后面跟着数字时，才算呼号
            next_is_nato_or_num = (i + 1 < len(words)) and (
                is_nato(words[i + 1]) or is_number_or_modifier(words[i + 1])
            )
            if next_is_nato_or_num:
                is_callsign_start = True

        if is_callsign_start:
            labels[i] = "B-CALLSIGN"
            j = 1
            # 吞噬后续：数字、NATO 音标、或其他航司词 (e.g., Jump Run)
            while (i + j) < len(words):
                next_w = words[i + j].lower()
                if is_number_or_modifier(next_w) or is_nato(next_w) or (next_w in AIRLINES):
                    labels[i + j] = "I-CALLSIGN"
                    j += 1
                else:
                    break
            i += j
            continue

        # 逻辑 2: 动作指令
        # 特殊处理 "Line Up", "Go Around"
        if w_lower == "line" and (i + 1 < len(words)) and words[i + 1].lower() == "up":
            labels[i] = "B-COMMAND"
            labels[i + 1] = "I-COMMAND"
            i += 2
            continue
        if w_lower == "go" and (i + 1 < len(words)) and words[i + 1].lower() == "around":
            labels[i] = "B-COMMAND"
            labels[i + 1] = "I-COMMAND"
            i += 2
            continue

        if w_lower in ATC_COMMANDS:
            labels[i] = "B-COMMAND"
            i += 1
            continue

        # 逻辑 3: 数值 (包含 Flight Level)
        # 如果遇到 "Flight Level"，直接作为 VALUE 的开始
        if w_lower == "flight" and (i + 1 < len(words)) and words[i + 1].lower() == "level":
            labels[i] = "B-VALUE"
            labels[i + 1] = "I-VALUE"
            j = 2
            # 继续吞噬后面的数字
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue

        if is_number_or_modifier(w_lower):
            # 只有还没被标为呼号的数字才算 Value
            if labels[i] == "O":
                labels[i] = "B-VALUE"
                j = 1
                while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                    labels[i + j] = "I-VALUE"
                    j += 1
                i += j
                continue

        # 逻辑 4: 航路点
        if w_upper in local_waypoints or (
            w_upper.isupper()
            and len(w_upper) >= 3
            and w_upper.isalpha()
            and not w_lower in ATC_COMMANDS
        ):
            # 简单的启发式：如果全是字母且大写，且不是指令，可能是航路点
            labels[i] = "B-WAYPOINT"
            i += 1
            continue

        i += 1

    return words, labels


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入目录: {INPUT_DIR}")
        return

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f'正在生成【规则标注】数据集，扫描 {len(json_files)} 个文件...')

    random.seed(42)
    random.shuffle(json_files)

    # 9:1 切分
    split_idx = int(len(json_files) * 0.9)
    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    def write_dataset(files, filename):
        out_path = os.path.join(OUTPUT_DIR, filename)
        count = 0
        with open(out_path, "w", encoding="utf-8") as out_f:
            for fpath in tqdm(files, desc=f"Writing {filename}"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    context = data.get("context", {})
                    for turn in data.get("dialogue", []):
                        text = turn.get("text", "")
                        if not text:
                            continue

                        tokens, tags = tag_entities(text, context)

                        # 只有包含有效信息的样本才保留
                        if any(t != "O" for t in tags):
                            out_f.write(json.dumps({"tokens": tokens, "ner_tags": tags}) + "\n")
                            count += 1
                except:
                    continue
        print(f'生成 {filename}: {count} 条样本')

    write_dataset(train_files, "train.json")
    write_dataset(test_files, "test.json")

    # 生成 Label Map
    label_list = [
        "O",
        "B-CALLSIGN",
        "I-CALLSIGN",
        "B-COMMAND",
        "I-COMMAND",
        "B-VALUE",
        "I-VALUE",
        "B-WAYPOINT",
        "I-WAYPOINT",
    ]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=4)

    print(f"\n规则标注数据集生成完毕！")
    print(f"路径: {OUTPUT_DIR}")
    print("关键修正：")
    print("   1. 增加了 KLM, JUMP, RUN 等呼号")
    print("   2. 增加了 Hotel, Golf 等 NATO 音标支持")
    print("   3. Flight Level 现在会被合并为 VALUE")
    print(
        "请修改 nlp_05_train_bert.py 的路径指向 'processed_data/ner_dataset_ultimate' 并重新训练！"
    )


if __name__ == "__main__":
    main()
