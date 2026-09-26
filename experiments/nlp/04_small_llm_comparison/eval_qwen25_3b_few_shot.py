# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/evaluate_qwen.py
# What it does: Same experiment with Qwen2.5-3B-Instruct.
# Known problems:
#   - Same comparability limits as the Phi-3 script.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from word2number import w2n
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TEST_FILE = "processed_data/ner_dataset_raw_split/test_raw.json"
OUTPUT_DIR = "qwen_evaluation_results"

# 对应 BERT 的标签体系 (合并 B/I 为实体类别)
LABEL_LIST = ["CALLSIGN", "COMMAND", "VALUE", "WAYPOINT"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 规则引擎 (生成 Ground Truth)
# Use the same labelling rules as training for comparison.
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
    "vista",
    "jet",
}
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
    "startup",
    "push",
    "start",
    "stop",
}
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
VALUE_MODIFIERS = {"flight", "level", "left", "right", "center", "centre"}
STOPWORDS = {
    "hello",
    "hi",
    "good",
    "morning",
    "afternoon",
    "evening",
    "day",
    "bye",
    "goodbye",
    "thank",
    "thanks",
    "roger",
    "wilco",
}


def is_number_or_modifier(w):
    return w.lower() in NUMBERS_AND_UNITS or w.isdigit() or w.lower() in VALUE_MODIFIERS


def is_nato(w):
    return w.lower() in ICAO_PHONETICS


def tag_entities(text, local_context):
    """Assign BIO labels using the existing vocabulary and context rules."""
    clean_text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    words = clean_text.split()
    labels = ["O"] * len(words)
    local_callsigns = set()
    if "callsigns" in local_context:
        for cs in local_context["callsigns"]:
            if isinstance(cs, str):
                local_callsigns.add(cs.split()[0].lower())
            elif isinstance(cs, dict) and "spoken" in cs:
                local_callsigns.add(cs["spoken"].split()[0].lower())
    local_waypoints = set(local_context.get("waypoints", []))

    i = 0
    while i < len(words):
        w_lower = words[i].lower()
        w_upper = words[i].upper()
        # CALLSIGN
        is_callsign_start = (w_lower in AIRLINES) or (w_lower in local_callsigns)
        if not is_callsign_start and is_nato(w_lower):
            next_is = (i + 1 < len(words)) and (
                is_nato(words[i + 1]) or is_number_or_modifier(words[i + 1])
            )
            if next_is:
                is_callsign_start = True
        if is_callsign_start:
            labels[i] = "B-CALLSIGN"
            j = 1
            while (i + j) < len(words):
                next_w = words[i + j].lower()
                if is_number_or_modifier(next_w) or is_nato(next_w) or (next_w in AIRLINES):
                    labels[i + j] = "I-CALLSIGN"
                    j += 1
                else:
                    break
            i += j
            continue
        # COMMAND
        if w_lower in ATC_COMMANDS:
            labels[i] = "B-COMMAND"
            i += 1
            continue
        if w_lower == "line" and (i + 1 < len(words)) and words[i + 1].lower() == "up":
            labels[i], labels[i + 1] = "B-COMMAND", "I-COMMAND"
            i += 2
            continue
        # VALUE
        if w_lower == "flight" and (i + 1 < len(words)) and words[i + 1].lower() == "level":
            labels[i], labels[i + 1] = "B-VALUE", "I-VALUE"
            i += 2
            j = 0
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue
        if is_number_or_modifier(w_lower):
            labels[i] = "B-VALUE" if labels[i] == "O" else labels[i]
            j = 1
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue
        # WAYPOINT
        if w_upper in local_waypoints or (
            w_upper.isupper()
            and len(w_upper) >= 3
            and w_upper.isalpha()
            and w_lower not in ATC_COMMANDS
            and w_lower not in STOPWORDS
        ):
            labels[i] = "B-WAYPOINT"
            i += 1
            continue
        i += 1
    return words, labels


def bio_to_entities(words, labels):
    """Group BIO-tagged tokens into entities."""
    entities = []
    current_type = None
    current_words = []
    for w, label in zip(words, labels):
        if label.startswith("B-"):
            if current_type:
                entities.append((current_type, " ".join(current_words).upper()))
            current_type = label[2:]
            current_words = [w]
        elif label.startswith("I-") and current_type == label[2:]:
            current_words.append(w)
        else:
            if current_type:
                entities.append((current_type, " ".join(current_words).upper()))
            current_type = None
            current_words = []
    if current_type:
        entities.append((current_type, " ".join(current_words).upper()))
    return entities


# 软匹配工具
def normalize(text):
    text = str(text).upper()
    mapping = {
        "ZERO": "0",
        "ONE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "DECIMAL": ".",
        "POINT": ".",
        "THOUSAND": "000",
        "HUNDRED": "00",
    }
    try:
        return str(w2n.word_to_num(text))
    except:
        pass
    words = text.split()
    res = "".join([mapping.get(w, w) for w in words])
    return re.sub(r"[^A-Z0-9]", "", res)


def is_soft_match(pred_val, gt_val):
    p, g = normalize(pred_val), normalize(gt_val)
    if not p or not g:
        return False
    return (p == g) or (p in g) or (g in p)


# 模型推理
def load_slm():
    print(f'加载 Qwen 模型: {MODEL_ID} ...')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    return tokenizer, model


def build_prompt_few_shot(atc_text):
    # Three examples in the prompt, using the same label rules.
    system = """You are an ATC entity extractor. Extract entities into JSON keys: CALLSIGN, COMMAND, VALUE, WAYPOINT.
Input: "lufthansa eight hotel romeo climb flight level one six zero"
Output: {"CALLSIGN": ["LUFTHANSA EIGHT HOTEL ROMEO"], "COMMAND": ["CLIMB"], "VALUE": ["FLIGHT LEVEL ONE SIX ZERO"]}
Input: "turn left heading two two zero"
Output: {"COMMAND": ["TURN"], "VALUE": ["LEFT", "TWO TWO ZERO"]}
Input: "contact praha radar one two zero decimal two seven five"
Output: {"COMMAND": ["CONTACT"], "VALUE": ["ONE TWO ZERO DECIMAL TWO SEVEN FIVE"], "WAYPOINT": ["PRAHA RADAR"]}
"""
    user = f'Input: "{atc_text}"\nOutput JSON:'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_llm(text):
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except:
        return {}


# 主程序
def main():
    # 准备数据
    print(f"读取测试数据并生成 Ground Truth...")
    data = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        raw_sessions = json.load(f)
        if not isinstance(raw_sessions, list):
            raw_sessions = [raw_sessions]
    for session in raw_sessions:
        ctx = session.get("context", {})
        dialogue = session.get("dialogue", []) if "dialogue" in session else [session]
        for turn in dialogue:
            txt = turn.get("text", "")
            if not txt:
                continue
            words, tags = tag_entities(txt, ctx)
            gt = bio_to_entities(words, tags)
            if gt:
                data.append({"text": txt, "ground_truth": gt})
    print(f"有效样本: {len(data)}")

    # 推理
    tokenizer, model = load_slm()
    y_true, y_pred = [], []
    correct_sentences = 0

    print(f"开始 Qwen 推理...")
    for item in tqdm(data):
        text = item["text"]
        gt_list = item["ground_truth"]  # [('COMMAND', 'CLIMB')]

        # 推理
        messages = build_prompt_few_shot(text)
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text_input], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                inputs.input_ids, max_new_tokens=128, temperature=0.1, do_sample=False
            )
        response = tokenizer.batch_decode(
            out_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        # 解析
        pred_json = extract_json_llm(response)
        pred_list = []
        if isinstance(pred_json, dict):
            for k, v in pred_json.items():
                vals = v if isinstance(v, list) else [v]
                for val in vals:
                    k_norm = k.upper()
                    if k_norm in LABEL_LIST:
                        pred_list.append((k_norm, str(val).upper()))

        # 构建混淆矩阵数据 (实体级对齐)
        matched_pred = set()
        sentence_error = False

        for gt_type, gt_val in gt_list:
            found = False
            for i, (p_type, p_val) in enumerate(pred_list):
                if i in matched_pred:
                    continue
                # 内容软匹配
                if is_soft_match(p_val, gt_val):
                    y_true.append(gt_type)
                    y_pred.append(p_type)  # 记录预测类型，看是否类型搞错
                    matched_pred.add(i)
                    found = True
                    break
            if not found:
                y_true.append(gt_type)
                y_pred.append("O")  # 漏报
                sentence_error = True

        for i, (p_type, p_val) in enumerate(pred_list):
            if i not in matched_pred:
                y_true.append("O")  # 误报
                y_pred.append(p_type)
                sentence_error = True

        if not sentence_error:
            correct_sentences += 1

    # 结果
    # 补充 "O" 到 LABEL_LIST 用于混淆矩阵
    CM_LABELS = ["O"] + LABEL_LIST

    print("\n" + "=" * 60)
    print(f"Sentence Accuracy: {correct_sentences / len(data):.2%}")
    print("-" * 60)
    report = classification_report(y_true, y_pred, labels=CM_LABELS, digits=4, zero_division=0)
    print(report)
    print("=" * 60)

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=CM_LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges", xticklabels=CM_LABELS, yticklabels=CM_LABELS
    )
    plt.title("Qwen SLM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qwen_confusion_matrix.png"))
    print(f"图表已保存: {OUTPUT_DIR}/qwen_confusion_matrix.png")


if __name__ == "__main__":
    main()
