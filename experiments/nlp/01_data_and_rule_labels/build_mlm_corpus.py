# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/atc_corpus.py
# What it does: Dumps every utterance into one text file for masked-language-model pretraining.
# Known problems:
#   - Includes the test transcripts, so the MLM pretraining saw test text.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 输入：之前 split_dataset_raw.py 生成的目录
INPUT_DIR = os.path.join(BASE_DIR, "processed_data", "ner_dataset_raw_split")
# 输出：纯文本语料库
OUTPUT_FILE = os.path.join(BASE_DIR, "processed_data", "atc_corpus.txt")


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入目录: {INPUT_DIR}")
        print("请先运行 split_dataset_raw.py！")
        return

    files_to_read = ["train_raw.json", "test_raw.json"]
    total_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for fname in files_to_read:
            fpath = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(fpath):
                print(f"跳过 {fname}(不存在)")
                continue

            print(f"正在读取 {fname}...")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for session in data:
                for turn in session.get("dialogue", []):
                    text = turn.get("text", "").strip()
                    # 过滤掉过短的无意义文本
                    if len(text) > 5:
                        out_f.write(text + "\n")
                        total_lines += 1

    print(f"语料提取完成！")
    print(f"保存路径: {OUTPUT_FILE}")
    print(f"总行数: {total_lines}")


if __name__ == "__main__":
    main()
