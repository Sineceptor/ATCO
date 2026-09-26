import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import json
import logging
import tensorflow as tf
from transformers import (
    BertTokenizerFast,
    TFBertForTokenClassification,
    DataCollatorForTokenClassification,

)
from datasets import Dataset

# Configuration

BASE_MODEL_PATH = str(ROOT / "models/bert-pretrained")

TRAIN_FILE = str(ROOT / "datasets/entities/train.jsonl")
TEST_FILE = str(ROOT / "datasets/entities/test.jsonl")

OUTPUT_DIR = str(ROOT / "outputs/training/bert-reference")


BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-5


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

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_data(filepath):
    """Read JSON, falling back to JSONL if needed."""
    if not os.path.exists(filepath):
        return None

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        try:

            data = json.load(f)
        except json.JSONDecodeError:

            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    if not data:
        return None

    formatted = {"id": [], "tokens": [], "ner_tags": []}
    for idx, item in enumerate(data):
        tag_ids = [LABEL2ID.get(t, 0) for t in item["ner_tags"]]
        formatted["id"].append(str(idx))
        formatted["tokens"].append(item["tokens"])
        formatted["ner_tags"].append(tag_ids)
    return Dataset.from_dict(formatted)


def main():

    logging.info("正在加载训练数据...")
    train_ds = load_data(TRAIN_FILE)
    test_ds = load_data(TEST_FILE)

    if not train_ds:
        logging.error("无法加载数据！请确保先运行了 training/prepare_entities.py")
        return


    logging.info(f"加载底座: {BASE_MODEL_PATH}")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_PATH)
    except:
        logging.warning("找不到预训练底座，将使用默认 bert-base-uncased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        BASE_MODEL_PATH_LOAD = "bert-base-uncased"
    else:
        BASE_MODEL_PATH_LOAD = BASE_MODEL_PATH

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
                    label_ids.append(-100)
                else:
                    label_ids.append(label[idx])
                prev_idx = idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    logging.info("处理数据中...")
    tf_train = train_ds.map(tokenize_align, batched=True).to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf"),
    )
    tf_test = test_ds.map(tokenize_align, batched=True).to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf"),
    )


    logging.info("初始化模型...")
    try:
        model = TFBertForTokenClassification.from_pretrained(
            BASE_MODEL_PATH_LOAD,
            num_labels=len(ID2LABEL),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            use_safetensors=False,
        )
    except TypeError:
        model = TFBertForTokenClassification.from_pretrained(
            BASE_MODEL_PATH_LOAD, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
        )


    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


    model.compile(optimizer=optimizer)


    logging.info(f'开始训练 (共 {EPOCHS} 轮)...')
    try:
        model.fit(tf_train, validation_data=tf_test, epochs=EPOCHS)
    except Exception as e:
        logging.error(f"训练再次中断: {e}")
        return


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging.info(f"保存模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("训练完成。")


if __name__ == "__main__":
    main()
