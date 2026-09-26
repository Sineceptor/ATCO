# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/Data_reduc.py
# What it does: Parses the ATCO2 1-hour corpus (XML + WAV), keeps English segments marked correct, cuts the audio into clips and writes one label JSON per clip.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import re
import json
import glob
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from collections import defaultdict
import math  # 导入 math 库用于处理无穷大

# Configuration
# Input dataset folder.
INPUT_DATA_DIR = "../ATCO2-ASRdataset-v1_beta/DATA"
# 脚本将在此处创建 processed_segments/audio 和 processed_segments/labels 文件夹
OUTPUT_BASE_DIR = "processed_data"


# 辅助函数：解析 .INFO 文件
def parse_info_file(info_file_path):
    """Read recording metadata and nearby callsigns from an .info file."""
    metadata = {}

    try:
        with open(info_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 提取 airport, frequency
            metadata["airport"] = (
                re.search(r"airport:\s*(.*?)\n", content, re.IGNORECASE).group(1).strip()
                if re.search(r"airport:\s*(.*?)\n", content, re.IGNORECASE)
                else None
            )
            metadata["frequency"] = (
                re.search(r"frequency:\s*(.*?)\n", content, re.IGNORECASE).group(1).strip()
                if re.search(r"frequency:\s*(.*?)\n", content, re.IGNORECASE)
                else None
            )

            # 提取呼号列表：从 'callsigns nearby:' 后面直到文件结束
            callsigns_match = re.search(r"callsigns nearby:\n(.*?)$", content, re.DOTALL)
            if callsigns_match:
                # 提取每行呼号，并只保留 ICAO 缩写部分（例如：从 "BLA131 : ..." 提取 "BLA131"）
                raw_callsigns = [
                    line.split(":")[0].strip()
                    for line in callsigns_match.group(1).splitlines()
                    if line.strip()
                ]
                metadata["callsigns_nearby"] = raw_callsigns
            else:
                metadata["callsigns_nearby"] = []

    except Exception as e:
        metadata["callsigns_nearby"] = []

    return metadata


# 辅助函数：清理 ASR 文本
def clean_asr_text(raw_text):
    """Remove transcript tags and collapse repeated whitespace."""
    # 匹配并移除 [#tag]...[/#tag] 格式的标签
    cleaned_text = re.sub(r"\[#.*?\]", "", raw_text)  # 移除起始标签，如 [#callsign]
    cleaned_text = re.sub(r"\[/.*?\]", "", cleaned_text)  # 移除结束标签，如 [/#callsign]

    # 清理多余空格并返回大写（ATC 约定俗成大写）
    return " ".join(cleaned_text.split()).upper()


# 主函数：处理所有数据
def process_data(input_dir, output_dir):
    # 创建输出目录
    audio_output_dir = os.path.join(output_dir, "audio")
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # 统计变量
    all_segments_data = []
    total_segments_count = 0
    passed_segments_count = 0
    total_duration_sec = 0.0
    speaker_counts = defaultdict(int)

    # 新增：用于跟踪最大和最小语段时长
    max_segment_duration_sec = 0.0
    min_segment_duration_sec = float("inf")  # 初始化为无穷大

    # 找到所有的 XML 文件，并获取唯一的 recording_id
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    recording_ids = [os.path.basename(f).replace(".xml", "") for f in xml_files]

    print(f'找到 {len(recording_ids)} 个原始录音文件进行处理...')

    for recording_id in recording_ids:
        xml_path = os.path.join(input_dir, f"{recording_id}.xml")
        wav_path = os.path.join(input_dir, f"{recording_id}.wav")
        info_path = os.path.join(input_dir, f"{recording_id}.info")

        if not os.path.exists(wav_path):
            print(f"Warning: WAV file not found for {recording_id}. Skipping.")
            continue

        # 解析 .info 文件
        info_metadata = parse_info_file(info_path)

        # 加载原始 WAV 文件
        try:
            audio = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(f"Error loading audio file {wav_path}. Skipping. Error: {e}")
            continue

        # 解析 .xml 文件
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_path}. Skipping. Error: {e}")
            continue

        segment_counter_in_file = 0

        for segment in root.findall("segment"):
            total_segments_count += 1

            # 提取核心信息
            start_time = float(segment.find("start").text) * 1000  # 转换为毫秒
            end_time = float(segment.find("end").text) * 1000  # 转换为毫秒
            raw_text = segment.find("text").text

            speaker_label_element = segment.find("speaker_label")
            speaker_label = (
                speaker_label_element.text
                if speaker_label_element is not None
                else segment.find("speaker").text
            )

            # 质量过滤 (QC)
            tags = segment.find("tags")
            correct_transcript = int(tags.find("correct_transcript").text)
            non_english = int(tags.find("non_english").text)

            if correct_transcript == 1 and non_english == 0:
                # 语段合格，开始处理
                passed_segments_count += 1
                segment_id = f"{recording_id}_{segment_counter_in_file:04d}"
                duration_ms = end_time - start_time
                duration_sec = duration_ms / 1000.0

                # 新增：更新最大和最小持续时间
                max_segment_duration_sec = max(max_segment_duration_sec, duration_sec)
                min_segment_duration_sec = min(min_segment_duration_sec, duration_sec)

                total_duration_sec += duration_sec
                speaker_counts[speaker_label] += 1

                # 音频切分与保存
                segment_audio = audio[start_time:end_time]
                segment_audio_path = os.path.join(audio_output_dir, f"{segment_id}.wav")
                segment_audio.export(segment_audio_path, format="wav")

                # 文本格式化与结构整合
                asr_text_clean = clean_asr_text(raw_text)

                segment_record = {
                    "utt_id": segment_id,
                    "original_recording_id": recording_id,
                    "duration_sec": duration_sec,
                    "speaker_label": speaker_label,
                    "audio_file": os.path.basename(segment_audio_path),
                    "ASR_transcript_clean": asr_text_clean,
                    "NLP_tagged_text": raw_text,
                    "metadata": info_metadata,
                }
                all_segments_data.append(segment_record)

                # 保存语段标签为 JSON 文件
                with open(
                    os.path.join(label_output_dir, f"{segment_id}.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(segment_record, f, indent=4, ensure_ascii=False)

            segment_counter_in_file += 1

    # 统计结果
    stats = {
        "Total_Original_Recordings": len(recording_ids),
        "Total_Segments_Found": total_segments_count,
        "Total_Segments_Passed_QC": passed_segments_count,
        "Total_Duration_Hours": total_duration_sec / 3600.0,
        "Total_Duration_Seconds": total_duration_sec,
        "Average_Segment_Length_Sec": total_duration_sec / passed_segments_count
        if passed_segments_count > 0
        else 0,
        # 新增统计结果
        "Max_Segment_Length_Sec": max_segment_duration_sec,
        "Min_Segment_Length_Sec": min_segment_duration_sec
        if min_segment_duration_sec != float("inf")
        else 0.0,
        "Speaker_Distribution": dict(speaker_counts),
        "Processed_Files_Location": os.path.abspath(output_dir),
    }

    # 保存总的统计数据
    with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    return stats


# 执行代码
if __name__ == "__main__":
    final_stats = process_data(INPUT_DATA_DIR, OUTPUT_BASE_DIR)

    print("\n--- 数据处理完成 ---\n")
    print("统计结果 (已包含最大和最小时长):")
    print(json.dumps(final_stats, indent=4, ensure_ascii=False))
