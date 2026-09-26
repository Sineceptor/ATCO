# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test bert.py
# What it does: Five hand-written sentences with hand-written labels, as a quick sanity check.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from transformers import pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "bert_model_strict")


# 外部测试数据 (Gold Standard)
# Hand-written examples; not a held-out evaluation set.
# 格式: {"text": "ASR出来的文本", "ground_truth": [("类别", "内容"), ...]}
# 注意：内容必须完全小写，去标点，与模型输入保持一致

EXTERNAL_TEST_DATA = [
    {
        "text": "singapore three eight two turn right heading zero nine zero",
        "ground_truth": {
            ("CALLSIGN", "singapore three eight two"),
            ("COMMAND", "turn"),
            ("VALUE", "right"),  # 有些模型可能会把 right 标为 VALUE 或 WAYPOINT，视训练数据而定
            ("COMMAND", "heading"),
            ("VALUE", "zero nine zero"),
        },
    },
    {
        "text": "united nine nine descend flight level two four zero",
        "ground_truth": {
            ("CALLSIGN", "united nine nine"),
            ("COMMAND", "descend"),
            ("VALUE", "flight level two four zero"),
        },
    },
    {
        "text": "cancel takeoff clearance delta five",
        "ground_truth": {
            ("COMMAND", "cancel"),
            ("COMMAND", "takeoff"),
            # clearance 可能是 O，视之前的标注规则而定
            ("CALLSIGN", "delta five"),
        },
    },
    {
        "text": "shamrock one two three contact ground one two one decimal nine",
        "ground_truth": {
            ("CALLSIGN", "shamrock one two three"),
            ("COMMAND", "contact"),
            ("WAYPOINT", "ground"),  # 有时候 ground 会被标为 waypoint
            ("VALUE", "one two one decimal nine"),
        },
    },
    {
        "text": "speedbird seven four seven climb two thousand",
        "ground_truth": {
            ("CALLSIGN", "speedbird seven four seven"),
            ("COMMAND", "climb"),
            ("VALUE", "two thousand"),
        },
    },
]


class StrictEvaluator:
    def __init__(self):
        print(f"加载模型: {MODEL_DIR}")
        if not os.path.exists(MODEL_DIR):
            print("模型不存在")
            exit(1)

        self.pipe = pipeline(
            "token-classification", model=MODEL_DIR, aggregation_strategy="simple", device=0
        )

    def normalize(self, text):
        """Convert to lowercase and strip surrounding whitespace."""
        return text.lower().strip()

    def evaluate(self, test_set):
        print(f'\n开始测试 {len(test_set)} 条外部数据...\n')

        total_correct = 0
        total_predicted = 0
        total_ground_truth = 0

        # 详细错误日志
        error_log = []

        for i, item in enumerate(test_set):
            text = item["text"]
            # 将 ground_truth 里的文本也标准化，确保比较公平
            ground_truth_set = set()
            for label, content in item["ground_truth"]:
                ground_truth_set.add((label, self.normalize(content)))

            # 模型推理
            predictions = self.pipe(text)

            # 构建预测集合
            predicted_set = set()
            for pred in predictions:
                p_label = pred["entity_group"]
                p_text = self.normalize(pred["word"])
                predicted_set.add((p_label, p_text))

            # 计算交集 (正确预测的部分)
            correct_matches = predicted_set.intersection(ground_truth_set)

            # 更新统计数据
            total_ground_truth += len(ground_truth_set)
            total_predicted += len(predicted_set)
            total_correct += len(correct_matches)

            # 如果不完全匹配，记录错误
            if correct_matches != ground_truth_set or correct_matches != predicted_set:
                missed = ground_truth_set - predicted_set
                wrong = predicted_set - ground_truth_set
                error_log.append(
                    {
                        "id": i + 1,
                        "text": text,
                        "missed": missed,  # 漏报
                        "wrong": wrong,  # 误报
                    }
                )

        # 计算最终指标
        print("=" * 40)
        print("最终评估报告 (Strict Matching)")
        print("=" * 40)

        # 精确率 (Precision): 预测出的实体里，有多少是对的？
        # 防止除以0
        precision = total_correct / total_predicted if total_predicted > 0 else 0

        # 召回率 (Recall): 真实存在的实体里，找出了多少？
        recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0

        # F1 Score: 综合分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"正确识别实体数: {total_correct}")
        print(f"真实实体总数:   {total_ground_truth}")
        print(f"模型预测总数:   {total_predicted}")
        print("-" * 40)
        print(f"Precision (查准率): {precision:.2%}")
        print(f"Recall    (查全率): {recall:.2%}")
        print(f"F1 Score  (综合分): {f1:.2%}")
        print("=" * 40)

        # 打印错误分析
        if error_log:
            print("\n错误样本分析 (Diff):")
            for err in error_log:
                print(f"\n样本 #{err['id']}: {err['text']}")
                if err["missed"]:
                    print(f"    漏识别 (Missed): {err['missed']}")
                if err["wrong"]:
                    print(f"    误识别 (Wrong):  {err['wrong']}")


if __name__ == "__main__":
    evaluator = StrictEvaluator()
    evaluator.evaluate(EXTERNAL_TEST_DATA)
