# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/merge.py
# What it does: Groups the per-clip label files by recording into one session JSON with context and an ordered dialogue.
# Known problems:
#   - Speaker defaults to 'Unknown' when the corpus gives no label.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import glob
import re
from collections import defaultdict
from tqdm import tqdm

# Configuration
# 输入：哪里去找那些碎片的 JSON？
INPUT_DIR = "./processed_data/labels"

# 输出：合并后的文件放在哪个新文件夹？
# (脚本会自动创建这个文件夹)
OUTPUT_DIR = "./processed_data/merged_sessions"


def get_file_info(filename):
    """Read the recording name and sequence number from a segment filename."""
    # 匹配规则：任意字符 + 下划线 + 数字 + .json
    match = re.match(r"(.*)_(\d+)\.json", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def main():
    # 检查输入
    if not os.path.exists(INPUT_DIR):
        print(f"找不到输入文件夹: {INPUT_DIR}")
        return

    # 创建新文件夹 (如果不存在)
    if not os.path.exists(OUTPUT_DIR):
        print(f"正在创建新文件夹: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    else:
        print(f"输出文件夹已存在: {OUTPUT_DIR}")

    # 扫描文件
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not json_files:
        print("输入文件夹是空的")
        return
    print(f'扫描到 {len(json_files)} 个碎片文件，准备合并...')

    # 分组 (Grouping)
    # 逻辑：把所有前缀相同的文件归为一组
    sessions = defaultdict(list)

    for fpath in tqdm(json_files, desc="正在分组"):
        filename = os.path.basename(fpath)
        base_name, seq_num = get_file_info(filename)

        if base_name is not None:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions[base_name].append((seq_num, data))
            except:
                continue

    print(f'共识别出 {len(sessions)} 个完整对话 session')

    # 排序与写入 (Sorting & Writing)
    success_count = 0

    for base_name, segment_list in tqdm(sessions.items(), desc="正在合并"):
        # 按序号 0, 1, 2... 排序，确保对话顺序正确
        segment_list.sort(key=lambda x: x[0])

        # 提取第一个片段的上下文作为整个会话的上下文
        first_data = segment_list[0][1]
        session_context = first_data.get("session_context", {})

        # 构建完整的大 JSON
        merged_record = {
            "session_id": base_name,
            "segment_count": len(segment_list),
            "context": session_context,  # 机场、呼号表等信息
            "dialogue": [],
        }

        # 填入每一句话
        for seq_id, data in segment_list:
            text = data.get("text", "") or data.get("ASR_transcript_clean", "")

            # 尝试获取说话人信息 (如果有)
            speaker = data.get("speaker_info", {}).get("label", "Unknown")

            merged_record["dialogue"].append(
                {
                    "seq": seq_id,
                    "speaker": speaker,
                    "text": text,
                    "audio": data.get("audio_filepath", "") or data.get("audio_file", ""),
                }
            )

        # 保存到新文件夹
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged_record, f, indent=4, ensure_ascii=False)

        success_count += 1

    print("\n" + "=" * 50)
    print(f"合并完成！")
    print(f"所有文件已保存在新文件夹: {OUTPUT_DIR}")
    print(f'共生成 {success_count} 个完整会话文件。')
    print("=" * 50)


if __name__ == "__main__":
    main()
