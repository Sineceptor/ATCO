# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/bert_analyzer_testset_tf.py
# What it does: Same, plus a stop-word list that forces words to 'O'.
# Known problems:
#   - The stop-word list was chosen from earlier evaluation logs.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import logging
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification

# Configuration
INPUT_FILE = "./processed_data/ner_dataset_ultimate/test.json"
OUTPUT_FILE = "./processed_data/analysis_report_v3.txt"
OUTPUT_JSONL = "./processed_data/final_analysis_v3.jsonl"
MODEL_PATH = "./bert_model_finetuned"  # Saved fine-tuned model.

# 拆分策略
SPLIT_TAGS = {"B-COMMAND", "B-ACTION", "B-ACT"}

# Set the listed stopwords to O as a heuristic.
# Noise terms selected from the earlier logs.
KNOWN_STOPWORDS = {
    # 问候/礼貌
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
    "sorry",
    "sir",
    "madam",
    # 常见虚词
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "with",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "will",
    "this",
    "that",
    "it",
    "we",
    "us",
    "me",
    "my",
    "your",
    "and",
    "or",
    "but",
    "so",
    "if",
    "then",
    # 噪音标签/元数据
    "ne",
    "hes",
    "spk",
    "ukn",
    "german",
    "french",
    "czech",
    "affirmation",
    "unk",
    "break",
    "correction",
}

# 白名单：这些词很容易被漏掉，强制召回
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
    "takeoff",
    "land",
    "vacate",
    "stop",
    "go",
    "expect",
    "monitor",
    "approach",
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

# 设置环境
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def refine_predictions(tokens, predicted_labels):
    """Apply vocabulary rules to the predicted labels."""
    refined_labels = []

    for word, label in zip(tokens, predicted_labels):
        word_lower = word.lower()
        new_label = label

        # 规则 1: 清洗 (Entity -> O)
        # 如果词在黑名单里 (如 hello)，不管模型预测什么，全部重置为 O
        if word_lower in KNOWN_STOPWORDS:
            new_label = "O"

        # 规则 2: 召回 (O -> Entity)
        # 如果模型漏掉了关键指令，强制改回来
        elif label == "O":
            if word_lower in KNOWN_VALUES:
                new_label = "B-VALUE"
            elif word_lower in KNOWN_COMMANDS:
                new_label = "B-COMMAND"

        refined_labels.append(new_label)

    return refined_labels


def merge_subwords_and_align(bert_tokens, bert_labels):
    merged_words = []
    merged_labels = []
    current_word = ""

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
    count = 0
    for label in labels:
        if label in SPLIT_TAGS:
            count += 1
    return count if count > 0 else 1


def split_commands(words, labels):
    segments = []
    current_text = []
    current_ents = []

    for word, label in zip(words, labels):
        if label in SPLIT_TAGS and current_text:
            segments.append({"text": " ".join(current_text), "entities": current_ents})
            current_text = []
            current_ents = []
        current_text.append(word)
        if label != "O":
            current_ents.append({"word": word, "type": label})

    if current_text:
        segments.append({"text": " ".join(current_text), "entities": current_ents})
    return segments


def load_dataset(filepath):
    data = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                return content
        except:
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
    logging.info(f"加载 NER 模型: {MODEL_PATH}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
        try:
            model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH, from_pt=True)
        except:
            model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH)

        raw_id2label = model.config.id2label
        id2label = {int(k): v for k, v in raw_id2label.items()}
    except Exception as e:
        logging.critical(f"模型加载失败: {e}")
        return

    data = load_dataset(INPUT_FILE)
    if not data:
        return

    logging.info(f'处理 {len(data)} 条数据...')
    analysis_results = []

    all_true = []
    all_pred = []

    for item in tqdm(data, desc="Processing"):
        original_tokens = item.get("tokens")
        text_str = " ".join(original_tokens) if original_tokens else (item.get("text") or "")
        if not text_str:
            continue

        inputs = tokenizer(
            text_str.lower(), return_tensors="tf", padding=True, truncation=True, max_length=128
        )
        outputs = model(inputs)
        predictions = tf.argmax(outputs.logits, axis=-1).numpy()[0]
        bert_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels_raw = [id2label.get(int(pid), "O") for pid in predictions]

        # 合并 Subword
        pred_words, pred_labels_merged = merge_subwords_and_align(bert_tokens, pred_labels_raw)

        # 规则修正
        refined_labels = refine_predictions(pred_words, pred_labels_merged)

        # 拆分
        segments = split_commands(pred_words, refined_labels)

        # 评估数据收集
        if item.get("ner_tags"):
            min_len = min(len(item["ner_tags"]), len(refined_labels))
            all_true.extend(item["ner_tags"][:min_len])
            all_pred.extend(refined_labels[:min_len])

        analysis_results.append(
            {
                "original_text": text_str,
                "segments_count": len(segments),
                "parsed_segments": segments,
            }
        )

    # 输出报告
    if all_true:
        print("\n" + "=" * 40)
        print("规则修正后的评估报告")
        print(classification_report(all_true, all_pred, digits=4))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for res in analysis_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    logging.info(f"完成！干净的结果已保存至: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
