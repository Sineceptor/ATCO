# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/划分训练集.py
# What it does: Shuffles all clips (seed 42) and writes train.jsonl / test.jsonl. Original filename was Chinese for 'split training set'.
# Known problems:
#   - Comments say 90/10 but TRAIN_RATIO is 0.8.
#   - Splits by clip, not by recording, so 92 recordings end up on both sides (see docs/evaluation.md).
#   - The character-vocabulary part is left over from an abandoned CTC plan.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import random
import glob

# 配置路径与比例
OUTPUT_BASE_DIR = "processed_data"
LABEL_BASE_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

# 划分比例：90% 训练，10% 测试
TRAIN_RATIO = 0.8

# 输出文件路径
VOCAB_FILE = os.path.join(OUTPUT_BASE_DIR, "char_vocab.json")
TRAIN_SPLIT_FILE = os.path.join(OUTPUT_BASE_DIR, "train.jsonl")
TEST_SPLIT_FILE = os.path.join(OUTPUT_BASE_DIR, "test.jsonl")


# 词汇表生成函数
def create_asr_vocab_and_load_data(label_dir, vocab_path):
    all_chars = set()
    all_data = []

    # 扫描所有 JSON 标签文件
    label_files_path = glob.glob(os.path.join(label_dir, "*.json"))

    if not label_files_path:
        print(f"错误：未找到任何 JSON 标签文件在 {label_dir}")
        return all_data

    print(f'正在加载 {len(label_files_path)} 个语段文件并提取词汇...')
    for fname in label_files_path:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("ASR_transcript_clean", "")
                # 收集所有出现的字符
                all_chars.update(list(text))
                all_data.append(data)
        except Exception as e:
            print(f'警告: 解析文件 {fname} 失败: {e}')

    # 定义 CTC 模型核心特殊标记
    # <pad> 必须在 ID 0
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

    # 词汇表列表：特殊标记 + 排序后的英文字符/符号
    vocab_list = special_tokens + sorted(list(all_chars))
    vocab_dict = {char: i for i, char in enumerate(vocab_list)}

    # 保存词汇表
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=4, ensure_ascii=False)

    print(f'词汇表已生成: {len(vocab_dict)} 个 token -> {vocab_path}')
    return all_data


# 二分数据集划分函数
def create_splits(all_data):
    """Split the dataset into training and test subsets."""
    random.seed(42)  # Fix the random seed for this sampling step.
    random.shuffle(all_data)

    total_size = len(all_data)
    train_size = int(TRAIN_RATIO * total_size)

    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    def save_jsonl(filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            for record in data:
                # 保持字段精简
                simplified_record = {
                    "audio_file": record["audio_file"],
                    "duration_sec": record["duration_sec"],
                    "ASR_transcript_clean": record["ASR_transcript_clean"],
                    "utt_id": record["utt_id"],
                }
                f.write(json.dumps(simplified_record, ensure_ascii=False) + "\n")

    save_jsonl(TRAIN_SPLIT_FILE, train_data)
    save_jsonl(TEST_SPLIT_FILE, test_data)

    print(f"数据集划分完成 (二分法 9:1)：")
    print(f"   总语段数: {total_size}")
    print(f'   训练集 (80%): {len(train_data)} 个 -> {TRAIN_SPLIT_FILE}')
    print(f'   测试集 (20%): {len(test_data)} 个 -> {TEST_SPLIT_FILE}')


# 执行
if __name__ == "__main__":
    print("---  ASR 预处理：词汇表生成与二分划分 ---")

    # 获取脚本所在目录，确保路径正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 步骤 1: 生成词汇表并加载数据
    dataset_data = create_asr_vocab_and_load_data(LABEL_BASE_DIR, VOCAB_FILE)

    # 步骤 2: 划分数据
    if dataset_data:
        create_splits(dataset_data)
    else:
        print("划分失败，数据为空。请检查 labels 文件夹路径。")
