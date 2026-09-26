# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/train_bert_ner_tf.py
# What it does: TensorFlow/Keras version of the DistilBERT tagger (3 epochs, lr 2e-5). The PyTorch version used by the app is training/train_distilbert.py.
# Known problems:
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import logging
import tensorflow as tf
from transformers import (
    DistilBertTokenizerFast,  # 必须用 Fast 版本
    TFDistilBertForTokenClassification,
    DataCollatorForTokenClassification,
    create_optimizer,
)
from datasets import Dataset

# Configuration
TRAIN_FILE = "./processed_data/ner_dataset_ultimate/train.json"
TEST_FILE = "./processed_data/ner_dataset_ultimate/test.json"
OUTPUT_DIR = "./bert_model_finetuned"  # 训练好的新模型将保存在这里
BASE_MODEL = "distilbert-base-uncased"

# 训练参数 (显存不够可将 Batch 改为 8)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Label IDs must match the training data.
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

# 设置环境
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_data(filepath):
    """Read JSON, falling back to JSONL if needed."""
    data = []
    if not os.path.exists(filepath):
        logging.error(f"找不到文件: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        # 尝试作为整个 JSON 读取
        try:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict) and "data" in content:
                data = content["data"]
            else:
                data = [content]
        except json.JSONDecodeError:
            # 失败则尝试按行读取 (JSONL)
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        continue

    formatted = {"id": [], "tokens": [], "ner_tags": []}
    for idx, item in enumerate(data):
        if "tokens" in item and "ner_tags" in item:
            # 转换标签为 ID
            tag_ids = [LABEL2ID.get(t, 0) for t in item["ner_tags"]]
            formatted["id"].append(str(idx))
            formatted["tokens"].append(item["tokens"])
            formatted["ner_tags"].append(tag_ids)

    logging.info(f"从 {os.path.basename(filepath)} 加载了 {len(formatted['id'])} 条数据")
    return Dataset.from_dict(formatted)


def main():
    # 加载数据
    train_ds = load_data(TRAIN_FILE)
    test_ds = load_data(TEST_FILE)
    if not train_ds or not test_ds:
        return

    # 初始化 Tokenizer
    logging.info("初始化 Tokenizer (Fast)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    def tokenize_align(examples):
        tokenized = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_idx = None
            label_ids = []
            for idx in word_ids:
                if idx is None or idx == prev_idx:
                    label_ids.append(-100)  # 忽略 subwords
                else:
                    label_ids.append(label[idx])
                prev_idx = idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    logging.info("数据预处理 (分词与对齐)...")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

    tf_train = train_ds.map(tokenize_align, batched=True).to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )
    tf_test = test_ds.map(tokenize_align, batched=True).to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    # 训练设置
    num_steps = len(tf_train) * EPOCHS
    optimizer, _ = create_optimizer(
        init_lr=LEARNING_RATE, num_train_steps=num_steps, num_warmup_steps=0
    )

    # 强制不使用 safetensors，解决报错
    logging.info("加载模型权重...")
    model = TFDistilBertForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=False,
    )
    model.compile(optimizer=optimizer)

    # 开始训练
    logging.info("开始训练...")
    model.fit(tf_train, validation_data=tf_test, epochs=EPOCHS)

    # 保存
    logging.info(f"保存新模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logging.info("训练完成！请进行第二步操作。")


if __name__ == "__main__":
    main()
