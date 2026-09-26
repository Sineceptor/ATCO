import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import logging
import tensorflow as tf
from transformers import (
    BertTokenizerFast,
    TFBertForMaskedLM,
    DataCollatorForLanguageModeling,
    create_optimizer,
)
from datasets import load_dataset


TRAIN_TEXT_FILE = str(ROOT / "datasets/entities/corpus.txt")

OUTPUT_DIR = str(ROOT / "outputs/training/bert-pretrained")

BASE_MODEL = "bert-base-uncased"


# Reduce BATCH_SIZE to 8 or 4 if GPU memory runs out.
BATCH_SIZE = 16
EPOCHS = 40  # Number of training epochs.
LEARNING_RATE = 5e-5


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    if not os.path.exists(TRAIN_TEXT_FILE):
        logging.error("找不到语料文件！请先准备配置路径下的文本语料。")
        return


    logging.info(f"加载语料库: {TRAIN_TEXT_FILE}")
    dataset = load_dataset("text", data_files={"train": TRAIN_TEXT_FILE})


    logging.info(f"初始化 Tokenizer ({BASE_MODEL})...")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)

    def tokenize_function(examples):

        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    logging.info("正在分词...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="tf"
    )

    tf_dataset = tokenized_datasets["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )


    logging.info(f"加载模型: {BASE_MODEL}(MaskedLM)")
    num_train_steps = len(tf_dataset) * EPOCHS
    optimizer, _ = create_optimizer(
        init_lr=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=0,
    )


    model = TFBertForMaskedLM.from_pretrained(BASE_MODEL, use_safetensors=False)
    model.compile(optimizer=optimizer)


    logging.info(f'开始预训练 (共 {EPOCHS} 轮)...')
    logging.info("训练时间取决于硬件和数据量。")

    try:
        model.fit(tf_dataset, epochs=EPOCHS)
    except Exception as e:
        logging.error(f"训练中断: {e}")
        logging.info("提示: 如果是显存不足(OOM)，请将脚本开头的 BATCH_SIZE 调小。")
        return


    logging.info(f"保存预训练模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logging.info("预训练完成！")
    logging.info("Saved BERT pretraining output. Use it as the base model for train_bert.py.")


if __name__ == "__main__":
    main()
