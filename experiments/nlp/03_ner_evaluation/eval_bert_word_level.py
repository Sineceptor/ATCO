# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/evaluate bert finetuned.py
# What it does: Evaluates the MLM-pretrained BERT tagger. Produced the 0.9544 weighted F1 in results/original_2025/.
# Known problems:
#   - The reference labels are regenerated at evaluation time by a third rule version, not the one used for training.
#   - Despite the 'entity-level' output name it scores single words with B-/I- prefixes removed.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import logging
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForTokenClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
# Saved BERT model.
MODEL_PATH = "./bert_base_ner_final_clean"
# 测试集路径
TEST_RAW_PATH = "processed_data/ner_dataset_raw_split/test_raw.json"
# 结果输出目录
OUTPUT_DIR = "bert_entity_level_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 原始训练时的标签映射
ID2LABEL = {
    0: "O",
    1: "B-CALLSIGN",
    2: "I-CALLSIGN",
    3: "B-COMMAND",
    4: "I-COMMAND",
    5: "B-VALUE",
    6: "I-VALUE",
    7: "B-WAYPOINT",
    8: "I-WAYPOINT",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# 论文专用的实体级标签列表 (去掉了 B- 和 I-)
ENTITY_LABELS = ["O", "CALLSIGN", "COMMAND", "VALUE", "WAYPOINT"]

# 屏蔽 TensorFlow 干扰日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# 规则引擎 (Ground Truth Generator)
# 必须与训练数据生成逻辑完全一致
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
    "you",
    "your",
    "we",
    "us",
    "sorry",
    "sir",
    "madam",
    "mister",
    "roger",
    "wilco",
    "affirm",
    "negative",
    "hes",
    "spk",
    "ne",
    "unk",
    "ukn",
    "break",
    "correction",
    "standby",
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


# 核心辅助函数


def load_raw_and_tag(filepath):
    """Load transcripts and generate reference labels with the labelling rules."""
    if not os.path.exists(filepath):
        print(f"找不到文件: {filepath}")
        return []
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            for session in raw_data:
                context = session.get("context", {})
                for turn in session.get("dialogue", []):
                    text = turn.get("text", "")
                    if not text:
                        continue
                    tokens, tags = tag_entities(text, context)
                    # 只要包含至少一个实体就加入评估
                    if any(t != "O" for t in tags):
                        samples.append({"tokens": tokens, "ner_tags": tags})
        except Exception as e:
            print(f"数据读取错误: {e}")
    print(f'成功加载: {len(samples)} 条样本')
    return samples


def convert_bio_to_entity(label):
    """Remove the B- and I- prefixes from token labels."""
    if label == "O":
        return "O"
    return label.split("-")[1]


# 主评估逻辑


def evaluate():
    print(f'正在加载 BERT 模型: {MODEL_PATH} ...')
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = TFBertForTokenClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 准备数据
    test_data = load_raw_and_tag(TEST_RAW_PATH)
    if not test_data:
        return

    print("开始实体级评估 (Entity-Level Evaluation)...")

    true_labels_flat = []
    pred_labels_flat = []

    for item in tqdm(test_data):
        tokens = item["tokens"]
        true_tags = item["ner_tags"]  # List of B-XXX, I-XXX

        # Tokenize
        tokenized = tokenizer(
            tokens, is_split_into_words=True, return_tensors="tf", truncation=True, max_length=128
        )

        # Inference
        logits = model(tokenized).logits
        pred_ids = tf.argmax(logits, axis=-1).numpy()[0]

        # Alignment & Collapse
        word_ids = tokenized.word_ids(batch_index=0)
        prev_word_idx = None

        for i, word_idx in enumerate(word_ids):
            # 跳过特殊字符
            if word_idx is None:
                continue

            # Word-Level 过滤：同一个单词只评测第一个 Subword
            if word_idx != prev_word_idx:
                raw_true = true_tags[word_idx]
                raw_pred = ID2LABEL.get(pred_ids[i], "O")

                # Entity-Level 转换：去前缀
                true_labels_flat.append(convert_bio_to_entity(raw_true))
                pred_labels_flat.append(convert_bio_to_entity(raw_pred))

            prev_word_idx = word_idx

    # 输出报告
    print("\n" + "=" * 60)
    print("BERT 最终评估结果 (Entity-Level / Word-Level Aligned)")
    print("=" * 60)

    # Accuracy
    acc = accuracy_score(true_labels_flat, pred_labels_flat)
    print(f"Accuracy: {acc:.4%}")
    print("-" * 60)

    # Classification Report
    print(
        classification_report(
            true_labels_flat, pred_labels_flat, labels=ENTITY_LABELS, digits=4, zero_division=0
        )
    )
    print("=" * 60)

    # 混淆矩阵可视化
    print("生成混淆矩阵...")
    cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=ENTITY_LABELS)

    plt.figure(figsize=(10, 8))
    # Use the Blues colour map.
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=ENTITY_LABELS, yticklabels=ENTITY_LABELS
    )
    plt.xlabel("Predicted Labels (BERT)")
    plt.ylabel("True Labels (Ground Truth)")
    plt.title("BERT NER Confusion Matrix (Entity Level)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "bert_final_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")

    # 保存文本结果
    with open(os.path.join(OUTPUT_DIR, "bert_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(
            classification_report(
                true_labels_flat, pred_labels_flat, labels=ENTITY_LABELS, digits=4, zero_division=0
            )
        )


if __name__ == "__main__":
    evaluate()
