# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/atc inference.py
# What it does: Older evaluator of the same model (accuracy 0.5321 in results/original_2025/).
# Known problems:
#   - Its reference rules label every word after an airline name as callsign, and it scores sub-word pieces the model was never trained on. The low score is mostly an evaluator bug.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizerFast, TFBertForTokenClassification

# Configuration
# Saved BERT model.
MODEL_PATH = "./bert_base_ner_final_clean"
# 测试数据路径
TEST_FILE = "processed_data/ner_dataset_raw_split/test_raw.json"
# 输出结果目录
OUTPUT_DIR = "bert_evaluation_results"

# 标签映射 (必须与 atc inference.py 中的定义完全一致)
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
LABEL_LIST = list(ID2LABEL.values())

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 规则引擎 (用于生成 Ground Truth)
# Generate reference labels with rules, not independent human annotation.
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


def is_number_or_modifier(w):
    return w.lower() in NUMBERS_AND_UNITS or w.isdigit()


def tag_entities(text):
    clean_text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    words = clean_text.split()
    labels = ["O"] * len(words)
    i = 0
    while i < len(words):
        w_lower = words[i].lower()
        # CALLSIGN
        if w_lower in AIRLINES:
            labels[i] = "B-CALLSIGN"
            j = 1
            while (i + j) < len(words) and (
                words[i + j].lower() in NUMBERS_AND_UNITS or words[i + j].isalnum()
            ):
                labels[i + j] = "I-CALLSIGN"
                j += 1
            i += j
            continue
        # COMMAND
        if w_lower in ATC_COMMANDS:
            labels[i] = "B-COMMAND"
            i += 1
            continue
        # VALUE
        if w_lower == "flight" and (i + 1 < len(words)) and words[i + 1].lower() == "level":
            labels[i], labels[i + 1] = "B-VALUE", "I-VALUE"
            i += 2
            continue
        if is_number_or_modifier(w_lower):
            labels[i] = "B-VALUE" if labels[i] == "O" else labels[i]
            j = 1
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue
        # WAYPOINT (简单规则: 大写且长)
        if words[i].isupper() and len(words[i]) > 3 and w_lower not in ATC_COMMANDS:
            labels[i] = "B-WAYPOINT"
            i += 1
            continue
        i += 1
    return words, labels


# 评估核心逻辑
def evaluate_bert(model_path, test_file):
    print(f'加载模型: {model_path} ...')
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = TFBertForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 加载数据
    print(f"读取测试数据: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raw_data = [raw_data]

    all_true_labels = []
    all_pred_labels = []

    print("开始推理与评估...")
    for session in tqdm(raw_data):
        for turn in session.get("dialogue", []):
            text = turn.get("text", "").strip()
            if not text:
                continue

            # A. 生成 Ground Truth (Word Level)
            words, word_labels = tag_entities(text)

            # B. 模型推理 (Token Level)
            inputs = tokenizer(text, return_tensors="tf", truncation=True, max_length=128)
            logits = model(inputs).logits
            predictions = tf.argmax(logits, axis=-1).numpy()[0]  # [CLS, tok1, tok2, SEP]

            # C. 对齐 (Alignment)
            # word_ids() maps each token to its word.
            # Align reference word labels with the token predictions.
            tokenized_inputs = tokenizer(
                text, truncation=True, max_length=128, return_offsets_mapping=True
            )
            word_ids = tokenized_inputs.word_ids()

            for idx, word_id in enumerate(word_ids):
                # 跳过 [CLS], [SEP], [PAD]
                if word_id is None:
                    continue

                # 防止索引越界 (极少数情况)
                if word_id >= len(word_labels):
                    break

                # 获取真值
                true_tag = word_labels[word_id]

                # 获取预测值
                pred_id = predictions[idx]
                pred_tag = ID2LABEL.get(pred_id, "O")

                # Subwords may share a word label.
                # This version scores every token, including continuation pieces.
                all_true_labels.append(true_tag)
                all_pred_labels.append(pred_tag)

    # 计算指标
    print("\n计算统计指标...")

    # Accuracy
    acc = accuracy_score(all_true_labels, all_pred_labels)

    # Classification Report (P/R/F1)
    # 使用 sklearn 生成详细报告，包含 Macro/Weighted Avg
    report = classification_report(all_true_labels, all_pred_labels, digits=4, zero_division=0)

    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {acc:.2%}")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # 保存报告文本
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 混淆矩阵可视化
    print("生成混淆矩阵...")

    # 计算矩阵
    # 获取数据中实际出现过的标签，防止矩阵过大包含没出现的类别
    unique_labels = sorted(list(set(all_true_labels + all_pred_labels)))
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)

    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("BERT NER Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"混淆矩阵图表已保存至: {save_path}")
    # plt.show() # 如果在 Notebook 中运行可取消注释


if __name__ == "__main__":
    evaluate_bert(MODEL_PATH, TEST_FILE)
