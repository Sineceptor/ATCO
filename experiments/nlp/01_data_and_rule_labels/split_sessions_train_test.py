# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/split_dataset_raw.py
# What it does: 90/10 split at session level, seed 42.
# Known problems:
#   - The file list is not sorted, so the split depends on filesystem order.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import glob
import random
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 输入：原始合并后的 session 文件
INPUT_DIR = os.path.join(BASE_DIR, "processed_data", "merged_sessions")
# 输出：拆分后的原始数据 (不带标签)
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data", "ner_dataset_raw_split")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def main():
    # 扫描文件
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入目录: {INPUT_DIR}")
        print("请检查 processed_data/merged_sessions 是否存在！")
        return

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    total_files = len(json_files)
    print(f'扫描到 {total_files} 个原始 Session 文件...')

    if total_files == 0:
        print("目录为空，请检查路径。")
        return

    # 随机打乱与拆分
    random.seed(42)  # 固定随机种子
    random.shuffle(json_files)

    split_ratio = 0.9
    split_idx = int(total_files * split_ratio)

    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    print(f'拆分结果: 训练集 {len(train_files)} 个, 测试集 {len(test_files)} 个')

    # 保存函数
    def save_raw_split(files, filename):
        out_path = os.path.join(OUTPUT_DIR, filename)
        saved_count = 0

        combined_data = []
        for fpath in tqdm(files, desc=f"Saving {filename}"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    combined_data.append(data)  # 保留原始结构
                    saved_count += 1
            except Exception as e:
                print(f"读取失败 {fpath}: {e}")
                continue

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(combined_data, out_f, indent=2, ensure_ascii=False)

        print(f"已保存: {out_path}")

    # 执行保存
    save_raw_split(train_files, "train_raw.json")
    save_raw_split(test_files, "test_raw.json")

    print("\n拆分完成！缺失的文件夹 ner_dataset_raw_split 已生成。")
    print("数据划分已保存。！")


if __name__ == "__main__":
    main()
