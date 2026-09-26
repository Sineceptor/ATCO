# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: extract_english.py
# What it does: Sorts the separate ATCO2 language-ID dataset into English / Others folders. Nothing else in the project uses its output.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import shutil
from tqdm import tqdm

# Configuration
# Dataset root.
SOURCE_ROOT = "ATCO2-LIDdataset-v1_beta"

# 输出目录
OUTPUT_BASE_DIR = "processed_data/atco2_classified_full"

# 英语文件夹名称识别 (只要文件夹叫这些，里面的所有文件都被带走)
ENGLISH_FOLDER_NAMES = ["EN", "English", "EN-AU", "en", "eng"]

# 其他语言文件夹名称 (用于归类到 Others)
OTHER_FOLDER_NAMES = ["CZ", "FR", "GE", "DE", "CH", "SK", "IT", "ES", "RU", "FR-CH"]

# File extensions to copy; an empty list copies all file types.
# 如果只想提取特定后缀，可以写: ['.wav', '.csv', '.xml', '.txt', '.json']
TARGET_EXTENSIONS = []


def main():
    print(f"开始全量提取数据 (音频 + 配套文件)...")
    print(f"源目录: {os.path.abspath(SOURCE_ROOT)}")

    # 创建输出目录
    dir_en = os.path.join(OUTPUT_BASE_DIR, "English")
    dir_others = os.path.join(OUTPUT_BASE_DIR, "Others")

    os.makedirs(dir_en, exist_ok=True)
    os.makedirs(dir_others, exist_ok=True)

    count_en = 0
    count_other = 0

    # 递归遍历所有目录
    for root, dirs, files in os.walk(SOURCE_ROOT):
        current_folder = os.path.basename(root)

        # 判断当前文件夹是英语还是其他
        target_dir = None
        if current_folder in ENGLISH_FOLDER_NAMES:
            target_dir = dir_en
        elif current_folder in OTHER_FOLDER_NAMES:
            target_dir = dir_others

        # 如果不是目标文件夹 (例如是上层目录 LID_DEVEL_CZEN)，跳过
        if target_dir is None:
            continue

        # 获取父目录名，用于给文件加前缀
        # 例如: 从 LID_DEVEL_CZEN/EN 提取文件
        # parent_folder 就是 "LID_DEVEL_CZEN"
        parent_folder = os.path.basename(os.path.dirname(root))

        # 遍历文件夹内的【所有文件】
        for file in files:
            # 扩展名过滤 (如果配置了的话)
            if TARGET_EXTENSIONS:
                _, ext = os.path.splitext(file)
                if ext.lower() not in TARGET_EXTENSIONS:
                    continue

            src_path = os.path.join(root, file)

            # 构建新文件名: 父目录_原文件名
            # 这样 audio.wav 和 audio.json 会变成:
            # LID_..._audio.wav 和 LID_..._audio.json
            # 它们在文件夹里依然会排在一起
            new_filename = f"{parent_folder}_{file}"
            dst_path = os.path.join(target_dir, new_filename)

            try:
                # copy2 会保留文件的原始修改时间等元数据
                shutil.copy2(src_path, dst_path)

                if target_dir == dir_en:
                    count_en += 1
                else:
                    count_other += 1

            except Exception as e:
                print(f"复制失败 {file}: {e}")

    print("\n" + "=" * 40)
    print("全量提取完成！")
    print(f'英语数据 (English): {count_en} 个文件 -> {dir_en}')
    print(f'其他语言 (Others) : {count_other} 个文件 -> {dir_others}')
    print("=" * 40)
    print("提示：同名的音频和信息文件已通过前缀自动对齐。")
    print("   例如: 'FolderA_123.wav' 和 'FolderA_123.xml' 现在紧挨着。")


if __name__ == "__main__":
    main()
