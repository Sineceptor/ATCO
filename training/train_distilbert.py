import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np


try:
    import evaluate
except ImportError:
    print("缺少依赖库，请先运行: pip install seqeval evaluate")
    exit(1)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rule-labelled training data.
NER_DATA_DIR = str(ROOT / "datasets/entities")


MODEL_OUTPUT_DIR = str(ROOT / "outputs/training/distilbert")


BASE_MODEL = "distilbert-base-uncased"


def train_ner():
    print(f"脚本位置: {BASE_DIR}")
    print(f"数据源: {NER_DATA_DIR}")


    train_file = os.path.join(NER_DATA_DIR, "train.jsonl")
    test_file = os.path.join(NER_DATA_DIR, "test.jsonl")
    map_file = str(ROOT / "training/label_map.json")


    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"找不到训练数据！")
        print(f"   请先准备配置路径下的 NER 数据文件。")
        return


    with open(map_file, "r") as f:
        mapping = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
        label2id = mapping["label2id"]
        label_list = list(label2id.keys())


    print("正在加载数据集...")
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f'    训练集: {len(train_dataset)} 条')
    print(f'    测试集: {len(eval_dataset)} 条')


    print(f"加载 BERT 底座: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], 0))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True)


    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)


        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )


    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n开始训练 (基于强规则数据)...")
    trainer.train()

    print("\n最终测试集评估:")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=4))

    print(f"\n训练完成！模型保存至: {MODEL_OUTPUT_DIR}")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    train_ner()
