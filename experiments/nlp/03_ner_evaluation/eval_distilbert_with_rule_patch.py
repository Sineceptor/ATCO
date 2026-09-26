# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/bert_analyzer_testset.py
# What it does: Evaluates DistilBERT after patching its 'O' predictions with keyword lists, and splits instructions at each command.
# Known problems:
#   - Circular: model + rules scored against rule-made labels.
#   - Alignment truncates to the shorter sequence.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification

# Configuration
INPUT_FILE = "./processed_data/ner_dataset_ultimate/test.json"
OUTPUT_FILE = "./processed_data/testset_evaluation_report.txt"
OUTPUT_JSONL = "./processed_data/final_analysis_refined.jsonl"
MODEL_PATH = "./bert_model_strict"

# 拆分策略：遇到这些标签时，判定为一条新指令的开始
SPLIT_TAGS = {"B-COMMAND", "B-ACTION", "B-ACT"}

# 屏蔽繁杂日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# 核心规则库：用于修正模型漏检的词
KNOWN_COMMANDS = {
    "climb",
    "descend",
    "maintain",
    "contact",
    "report",
    "turn",
    "cleared",
    "hold",
    "line",
    "taxi",
    "squawk",
    "pushback",
    "approved",
    "cancel",
    "proceed",
    "cross",
    "takeoff",
    "land",
    "vacate",
    "stop",
    "go",
    "expect",
    "monitor",
}
KNOWN_VALUES = {
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
    "flight",
    "level",
    "heading",
    "degrees",
    "knots",
    "feet",
    "runway",
    "qnh",
    "left",
    "right",
    "mach",
}


def refine_predictions(tokens, predicted_labels):
    """Apply vocabulary rules to the predicted labels."""
    refined_labels = []

    for word, label in zip(tokens, predicted_labels):
        word_lower = word.lower()
        new_label = label

        # 规则修正逻辑
        if label == "O":
            if word_lower in KNOWN_VALUES:
                new_label = "B-VALUE"
            elif word_lower in KNOWN_COMMANDS:
                new_label = "B-COMMAND"

        refined_labels.append(new_label)

    return refined_labels


def merge_subwords_and_align(bert_tokens, bert_labels):
    """Join WordPiece fragments and align them to words."""
    merged_words = []
    merged_labels = []

    current_word = ""
    # Use the first subword label for each word.

    for token, label in zip(bert_tokens, bert_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                merged_words.append(current_word)

            current_word = token
            merged_labels.append(label)

    if current_word:
        merged_words.append(current_word)

    return merged_words, merged_labels


def count_segments(labels):
    """Count command starts in the predicted labels."""
    count = 0
    has_command = False
    for label in labels:
        if label in SPLIT_TAGS:
            count += 1
            has_command = True
    return count if has_command else 1


def split_commands(words, labels):
    """Split on B-COMMAND labels."""
    segments = []
    current_segment_text = []
    current_segment_entities = []

    for word, label in zip(words, labels):
        # 拆分点检测
        if label in SPLIT_TAGS and current_segment_text:
            segments.append(
                {"text": " ".join(current_segment_text), "entities": current_segment_entities}
            )
            current_segment_text = []
            current_segment_entities = []

        current_segment_text.append(word)

        if label != "O":
            current_segment_entities.append({"word": word, "type": label})

    if current_segment_text:
        segments.append(
            {"text": " ".join(current_segment_text), "entities": current_segment_entities}
        )

    return segments


def load_dataset(filepath):
    data = []
    if not os.path.exists(filepath):
        logging.error(f"找不到文件: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                return content
        except json.JSONDecodeError:
            pass
        f.seek(0)
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def main():
    # 加载模型
    logging.info(f"加载 NER 模型: {MODEL_PATH}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
        try:
            model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH, from_pt=True)
        except:
            model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH)

        # 修正 id2label 键类型
        raw_id2label = model.config.id2label
        id2label = {int(k): v for k, v in raw_id2label.items()}

    except Exception as e:
        logging.critical(f"模型加载失败: {e}")
        return

    # 读取数据
    data = load_dataset(INPUT_FILE)
    if not data:
        return
    logging.info(f'准备处理 {len(data)} 条数据...')

    # 收集评测指标
    all_true_tags = []
    all_pred_tags = []
    true_segment_counts = []
    pred_segment_counts = []

    analysis_results = []

    # 推理循环
    for item in tqdm(data, desc="Processing"):
        # 获取输入
        original_tokens = item.get("tokens")
        true_tags = item.get("ner_tags")

        if not original_tokens or not isinstance(original_tokens, list):
            # 兼容非 token 列表格式的数据
            text = item.get("text") or item.get("command")
            if not text:
                continue
            text_str = text
            # 如果没有真实标签，造一个假的空标签用于跑通流程（不计入评估）
            has_ground_truth = False
        else:
            text_str = " ".join(original_tokens)
            has_ground_truth = True

        # BERT 推理
        inputs = tokenizer(
            text_str.lower(), return_tensors="tf", padding=True, truncation=True, max_length=128
        )
        outputs = model(inputs)

        predictions = tf.argmax(outputs.logits, axis=-1).numpy()[0]
        bert_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels_raw = [id2label.get(int(pid), "O") for pid in predictions]

        # 后处理 1: 合并 Subwords
        pred_words, pred_labels_merged = merge_subwords_and_align(bert_tokens, pred_labels_raw)

        # 后处理 2:  规则修正 (解决 'O' 标签问题)
        refined_labels = refine_predictions(pred_words, pred_labels_merged)

        # 后处理 3: 指令拆分
        split_segments = split_commands(pred_words, refined_labels)

        # 收集评估数据 (如果有真实标签)
        if has_ground_truth:
            # 截断对齐
            min_len = min(len(true_tags), len(refined_labels))
            all_true_tags.extend(true_tags[:min_len])
            all_pred_tags.extend(refined_labels[:min_len])

            t_count = count_segments(true_tags)
            p_count = count_segments(refined_labels)
            true_segment_counts.append(t_count)
            pred_segment_counts.append(p_count)

        # 构造结果
        result_item = {
            "original_text": text_str,
            "segments_count": len(split_segments),
            "parsed_segments": split_segments,
        }
        analysis_results.append(result_item)

    # 打印评估报告
    if all_true_tags:
        print("\n" + "=" * 50)
        print("【BERT NER 模型最终评估报告 (含规则修正)】")
        print("=" * 50)

        # A. 实体指标
        report = classification_report(all_true_tags, all_pred_tags, digits=4)
        print("\n1. 实体标签分类性能:")
        print(report)

        # B. 拆分指标
        seg_acc = accuracy_score(true_segment_counts, pred_segment_counts)
        print(f"\n2. 指令拆分准确率: {seg_acc:.2%}")

        # 保存报告
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
            f.write(f"\nCommand Split Accuracy: {seg_acc:.4f}\n")

    # 保存结果 JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for res in analysis_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    logging.info(f"完成！结果保存在: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
