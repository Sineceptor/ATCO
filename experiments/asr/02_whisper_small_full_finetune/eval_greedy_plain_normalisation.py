# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR2/quick_test.py
# What it does: Greedy decoding on the test set with plain scoring (uppercase, no punctuation). This is the honest evaluator; its archived log reconstructs to 27.19% WER.
# Known problems:
#   - Clips that fail to load are skipped silently.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import torch
import jiwer
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configuration
# Saved model folder.
MODEL_PATH = "whisper_atco_raw_V2"

# 测试集路径
TEST_JSONL = "./processed_data/test.jsonl"

# 音频文件夹路径
AUDIO_ROOT = "./processed_data/audio"

# 结果保存文件
REPORT_FILE = "test_report.txt"


class ATCEvaluator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载模型到 {self.device.upper()}...")
        print(f"   路径: {model_path}")

        try:
            self.processor = WhisperProcessor.from_pretrained(
                model_path, language="English", task="transcribe"
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()  # 切换到评估模式 (关闭 Dropout 等)
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)

        # 定义文本标准化器 (转大写，去标点，去多余空格)
        # 这是计算 WER 的标准步骤，防止因为 "." 或 "," 导致判错
        self.normalizer = jiwer.Compose(
            [
                jiwer.ToUpperCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )

    def load_test_data(self):
        print(f"正在读取测试集: {TEST_JSONL}")
        data = []
        if not os.path.exists(TEST_JSONL):
            print("找不到测试文件")
            return []

        with open(TEST_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f'   -> 共 {len(data)} 条测试样本')
        return data

    def evaluate(self):
        test_data = self.load_test_data()
        if not test_data:
            return

        predictions = []
        references = []

        # 用于记录详细日志
        log_buffer = []

        print("\n开始批量推理 (请耐心等待)...")

        # 使用 tqdm 显示进度条
        for item in tqdm(test_data, unit="条"):
            # 获取路径和真值
            filename = os.path.basename(item["audio_file"])
            audio_path = os.path.join(AUDIO_ROOT, filename)
            ground_truth = item["ASR_transcript_clean"]

            # 检查文件
            if not os.path.exists(audio_path):
                # 尝试去 DATA 找 (容错)
                fallback = os.path.join("./DATA/Processed_Dataset/clips", filename)
                if os.path.exists(fallback):
                    audio_path = fallback
                else:
                    continue  # 找不到文件跳过，不计入成绩

            try:
                # 读取音频
                audio, sr = sf.read(audio_path, dtype="float32")

                # 预处理
                input_features = self.processor(
                    audio, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(self.device)

                # 模型生成 (Greedy Search)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features, language="en", max_new_tokens=128
                    )

                # 解码文本
                transcription = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # 标准化 (重要！)
                ref_norm = self.normalizer(ground_truth)
                hyp_norm = self.normalizer(transcription)

                # 存入列表
                references.append(ref_norm)
                predictions.append(hyp_norm)

                # 如果预测错了，记录下来方便分析
                if ref_norm != hyp_norm:
                    log_buffer.append(
                        f"[错误] File: {filename}\nRef: {ref_norm}\nHyp: {hyp_norm}\n"
                    )
                else:
                    # 如果想看对的，也可以解开下面这行
                    # log_buffer.append(f"[正确] File: {filename}\nRef: {ref_norm}\n")
                    pass

            except Exception as e:
                print(f'推理异常: {filename} - {e}')

        # 计算最终指标
        print("\n正在计算准确率...")

        if len(references) > 0:
            # 计算 WER (词错误率)
            wer = jiwer.wer(references, predictions)
            # 计算 CER (字符错误率)
            cer = jiwer.cer(references, predictions)

            # 估算准确率 (Accuracy = 1 - WER)
            # 注意：WER 可能大于 100%，所以准确率最低限制为 0%
            accuracy = max(0, 1 - wer)

            # 生成报告字符串
            summary = (
                f"\n{'=' * 20} 最终评估报告 {'=' * 20}\n"
                f"样本总数: {len(references)}\n"
                f"----------------------------------------\n"
                f"词错误率 (WER): {wer:.2%}  (越低越好)\n"
                f"字错误率 (CER): {cer:.2%}  (越低越好)\n"
                f"----------------------------------------\n"
                f"估算准确率 (Acc): {accuracy:.2%} (越高越好)\n"
                f"{'=' * 54}\n"
            )

            print(summary)

            # 保存到文件
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                f.write(summary + "\n")
                f.write("=== 详细错误分析 (仅展示不一致的样本) ===\n\n")
                for log in log_buffer:
                    f.write(log + "\n" + "-" * 50 + "\n")

            print(f"详细错误报告已保存至: {REPORT_FILE}")
            print(f"   (请打开该文件查看具体是哪些词听错了)")

        else:
            print("没有产生有效的预测结果 (可能是路径不对导致没有读取到任何音频)。")


if __name__ == "__main__":
    evaluator = ATCEvaluator(MODEL_PATH)
    evaluator.evaluate()
