# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test_bert_full.py
# What it does: Demo: entities -> intent -> standardised text (airline to ICAO code, spoken digits to numerals, e.g. FL300).
# Known problems:
#   - No metrics.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import random
import re
import torch
from transformers import pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Saved model folder.
MODEL_DIR = os.path.join(BASE_DIR, "bert_model_strict")
TEST_FILE = os.path.join(BASE_DIR, "processed_data", "ner_dataset_ultimate", "test.json")


# 模拟测试数据

EXTERNAL_DATA = [
    {
        "text": "qantas four five six climb flight level three zero zero",
        "desc": "标准指令 (Standard)",
    },
    {"text": "turn right heading two seven zero", "desc": "缺呼号 (Missing Callsign)"},
    {"text": "descend two thousand", "desc": "模糊指令 (Ambiguous)"},
    {
        "text": "singapore eight eight one contact melbourne centre one two three decimal four",
        "desc": "频率移交 (Frequency Change)",
    },
]


# 纠错与标准化工具 (Corrector Utils)


class AtcCorrector:
    def __init__(self):
        # 数字映射
        self.num_map = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "zero": "0",
            "tree": "3",
            "niner": "9",
            "fife": "5",
            "thousand": "000",
            "hundred": "00",
        }
        # 航司映射 (Airline -> ICAO Code)
        self.airline_map = {
            "jetstar": "JST",
            "qantas": "QFA",
            "velocity": "VOZ",
            "virgin": "VOZ",
            "rex": "RXA",
            "united": "UAL",
            "singapore": "SIA",
            "delta": "DAL",
            "china": "CCA",
            "air": "CCA",  # 简化处理
            "klm": "KLM",
            "speedbird": "BAW",
            "shamrock": "EIN",
        }
        # 术语缩写
        self.term_map = {
            "flight level": "FL",
            "degrees": "",
            "knots": "kts",
            "decimal": ".",
            "point": ".",
        }

    def text_to_digit(self, text):
        """Join spoken digit words, for example one two three becomes 123."""
        words = text.lower().split()
        res = []
        for w in words:
            if w in self.num_map:
                res.append(self.num_map[w])
            elif w in self.term_map:
                res.append(self.term_map[w])
            else:
                res.append(w)
        return "".join(res).upper()

    def standardize_callsign(self, text):
        """Convert known airline names and spoken digits to a callsign."""
        words = text.lower().split()
        icao = words[0].upper()
        # 尝试匹配航司代码
        for name, code in self.airline_map.items():
            if name in words:
                icao = code
                break

        # 提取数字部分
        nums = []
        for w in words:
            if w in self.num_map:
                nums.append(self.num_map[w])

        return f"{icao}{''.join(nums)}"

    def standardize(self, entities):
        """Normalise the extracted entity text."""
        corrected_parts = []

        for ent in entities:
            label = ent["entity_group"]
            word = ent["word"]

            if label == "CALLSIGN":
                corrected_parts.append(self.standardize_callsign(word))
            elif label == "VALUE":
                # 特殊处理 FL
                val = self.text_to_digit(word)
                if "flight level" in word.lower() and not val.startswith("FL"):
                    val = "FL" + val
                corrected_parts.append(val)
            elif label == "COMMAND":
                corrected_parts.append(word.upper())
            elif label == "WAYPOINT":
                corrected_parts.append(word.upper())

        return " ".join(corrected_parts)


class BertEvaluator:
    def __init__(self):
        print(f'加载模型: {MODEL_DIR} ...')
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "token-classification", model=MODEL_DIR, aggregation_strategy="simple", device=device
        )
        self.corrector = AtcCorrector()

    def analyze_intent(self, entities):
        commands = [e["word"].lower() for e in entities if e["entity_group"] == "COMMAND"]
        if not commands:
            return "Unknown"

        mapping = {
            "climb": "CLIMB",
            "descend": "DESCEND",
            "turn": "VECTOR",
            "heading": "VECTOR",
            "contact": "HANDOFF",
            "cleared": "CLEARED",
        }

        for cmd in commands:
            if cmd in mapping:
                return mapping[cmd]
        return "INSTRUCTION"

    def check_logic(self, entities):
        """Apply the existing range and format checks."""
        types = [e["entity_group"] for e in entities]
        warnings = []
        if "COMMAND" in types and "VALUE" not in types and "WAYPOINT" not in types:
            warnings.append("缺少数值")
        if "CALLSIGN" not in types:
            warnings.append("缺少呼号")
        return warnings

    def run_test(self):
        # 混合数据
        test_samples = EXTERNAL_DATA.copy()

        # 从测试集随机抽 2 条
        if os.path.exists(TEST_FILE):
            with open(TEST_FILE, "r") as f:
                lines = f.readlines()
                if lines:
                    random_lines = random.sample(lines, min(2, len(lines)))
                    for line in random_lines:
                        data = json.loads(line)
                        test_samples.append(
                            {"text": " ".join(data["tokens"]), "desc": "随机测试集样本"}
                        )

        print(f'开始对比测试 ({len(test_samples)} 条数据)...\n')

        for i, sample in enumerate(test_samples):
            text = sample["text"]
            desc = sample["desc"]

            # 模型预测
            entities = self.pipe(text)

            # 意图分析
            intent = self.analyze_intent(entities)

            # 标准化 (纠错核心)
            standardized_text = self.corrector.standardize(entities)

            # 逻辑警告
            logic_warns = self.check_logic(entities)
            status_icon = "No missing fields detected" if not logic_warns else "Missing fields"

            # 打印对比报告
            print("=" * 60)
            print(f"Case {i + 1}: {desc}")
            print("-" * 60)
            print(f'[原始语音]: "{text}"')
            print(f"[识别意图]: {intent}")

            # 纠错对比展示
            print("-" * 60)
            print(f"[智能纠错]:")
            print(f"   Before: {text}")
            print(f"   After : \033[1;32m{standardized_text}\033[0m")  # 绿色高亮

            if logic_warns:
                print(f"   Status: {status_icon} 警告: {', '.join(logic_warns)}")
            else:
                print(f'   Status: {status_icon} 完整')
            print("\n")


if __name__ == "__main__":
    tester = BertEvaluator()
    tester.run_test()
