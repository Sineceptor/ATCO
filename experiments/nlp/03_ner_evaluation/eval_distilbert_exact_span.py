# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test_bert_prediction.py
# What it does: Strict exact-match span precision/recall/F1 for DistilBERT against the stored rule labels.
# Known problems:
#   - Set-based matching collapses repeated entities.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import random
import torch
from transformers import pipeline
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved model folder.

MODEL_DIR = os.path.join(BASE_DIR, "bert_model_strict")

# 测试集路径 (规则标注数据)
TEST_FILE = os.path.join(BASE_DIR, "processed_data", "ner_dataset_ultimate", "test.json")


class BertEvaluator:
    def __init__(self):
        print(f"正在加载模型: {MODEL_DIR}...")
        if not os.path.exists(MODEL_DIR):
            print(f"找不到模型文件夹: {MODEL_DIR}")
            print("   请先运行训练脚本 nlp_05_train_bert.py")
            exit(1)

        device = 0 if torch.cuda.is_available() else -1
        # aggregation_strategy="simple" 是关键，它会自动合并 "Jet" + "star" -> "Jetstar"
        self.pipe = pipeline(
            "token-classification", model=MODEL_DIR, aggregation_strategy="simple", device=device
        )
        print("模型加载完毕！")

    def load_test_data(self):
        if not os.path.exists(TEST_FILE):
            print(f"找不到测试文件: {TEST_FILE}")
            return []

        data = []
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def reconstruct_ground_truth(self, tokens, tags):
        """Group the reference tokens and BIO labels into entity text."""
        entities = []
        current_word = []
        current_label = None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current_word:
                    entities.append((current_label, " ".join(current_word)))
                current_word = [token]
                current_label = tag[2:]  # 去掉 B-
            elif tag.startswith("I-"):
                if current_word and tag[2:] == current_label:
                    current_word.append(token)
                else:
                    # 标签断裂或错误的情况，强制开始新词
                    if current_word:
                        entities.append((current_label, " ".join(current_word)))
                    current_word = [token]
                    current_label = tag[2:]
            else:  # Tag is O
                if current_word:
                    entities.append((current_label, " ".join(current_word)))
                current_word = []
                current_label = None

        if current_word:
            entities.append((current_label, " ".join(current_word)))

        return entities

    def evaluate(self):
        data = self.load_test_data()
        if not data:
            return

        print(f'\n正在评估测试集 ({len(data)} 条样本)...')
        print("=" * 70)

        # 统计指标
        correct_count = 0
        total_pred_count = 0
        total_true_count = 0

        # 错误日志
        mistakes = []

        for sample in data:
            raw_text = " ".join(sample["tokens"])

            # 获取真实标签 (Ground Truth)
            true_entities = self.reconstruct_ground_truth(sample["tokens"], sample["ner_tags"])
            true_set = set((label, text.lower()) for label, text in true_entities)

            # 获取模型预测 (Prediction)
            preds = self.pipe(raw_text)
            pred_set = set()
            for p in preds:
                pred_set.add((p["entity_group"], p["word"].lower().strip()))

            # 对比
            intersection = true_set.intersection(pred_set)
            correct_count += len(intersection)
            total_pred_count += len(pred_set)
            total_true_count += len(true_set)

            # 记录错误 (如果没完全对)
            if true_set != pred_set:
                mistakes.append(
                    {"text": raw_text, "missed": true_set - pred_set, "wrong": pred_set - true_set}
                )

        # 计算 F1
        precision = correct_count / total_pred_count if total_pred_count else 0
        recall = correct_count / total_true_count if total_true_count else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        print(f"整体准确率报告 (Strict Mode):")
        print(f"   Precision (查准率): {precision:.2%}  (预测出的实体有多少是对的)")
        print(f"   Recall    (查全率): {recall:.2%}  (原本的实体抓出了多少)")
        print(f"   F1 Score  (综合分): {f1:.2%}  (越高越好)")
        print("=" * 70)

        # 展示随机抽查结果
        print("\n随机抽查 5 个样本展示详情:")
        samples = random.sample(data, min(5, len(data)))

        for sample in samples:
            raw_text = " ".join(sample["tokens"])
            true_entities = self.reconstruct_ground_truth(sample["tokens"], sample["ner_tags"])

            print("-" * 70)
            print(f"文本: {raw_text}")
            print(f"真值: ", end="")
            for label, text in true_entities:
                print(f"[{label}] {text}  ", end="")
            print("\n预测: ", end="")

            preds = self.pipe(raw_text)
            if not preds:
                print("(无结果)", end="")
            for p in preds:
                print(f"[{p['entity_group']}] {p['word']} ({p['score']:.0%})  ", end="")
            print("\n")


def main():
    tester = BertEvaluator()
    tester.evaluate()


if __name__ == "__main__":
    main()
