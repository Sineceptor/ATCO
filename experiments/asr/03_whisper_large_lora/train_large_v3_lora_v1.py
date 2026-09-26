# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_improve.py
# What it does: Whisper-large-v3 in 4-bit with LoRA (r=32, q/v), lr 1e-3, 1000 steps, random Gaussian noise added on the fly.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# Set Hugging Face cache path
os.environ["HF_HOME"] = "./hf_cache"
# Disable OneDNN optimization logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import torch
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from torch.utils.data import Dataset

# Import PEFT (LoRA) and Quantization libraries
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

# Configuration
# Use Large-v3 model with LoRA. Low VRAM usage, but accuracy far exceeds Small
MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "atc_whisper_lora_v1_fixed"
TRAIN_FILE = "processed_data/train.jsonl"
TEST_FILE = "processed_data/test.jsonl"
AUDIO_ROOT = "./processed_data/audio"


def check_gpu():
    if not torch.cuda.is_available():
        print("Fatal Error: GPU not detected!")
        exit(1)
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")


# Dataset Class (Includes Data Augmentation)
class ATCInMemoryDataset(Dataset):
    def __init__(self, jsonl_path, processor, augment=False):
        self.processor = processor
        self.audio_root = AUDIO_ROOT
        self.cached_features = []
        self.augment = augment

        print(f"Loading: {jsonl_path}")
        if not os.path.exists(jsonl_path):
            return

        lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))

        print(f"Processing audio data...")
        for item in tqdm(lines):
            filename = os.path.basename(item["audio_file"])
            audio_path = os.path.join(self.audio_root, filename)
            # Standardize to uppercase to solve inconsistent spelling
            text = item["ASR_transcript_clean"].upper()

            if not os.path.exists(audio_path):
                continue

            # Only store path and text here, decoding and augmentation happen in __getitem
            self.cached_features.append({"path": audio_path, "text": text})

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        item = self.cached_features[idx]
        audio, sr = sf.read(item["path"], dtype="float32")

        # Audio Augmentation (Enabled only for training)
        if self.augment:
            # Randomly inject Gaussian noise to simulate radio static and prevent overfitting
            if np.random.rand() < 0.3:  # 30% probability to augment
                noise = np.random.randn(len(audio))
                # 005 is noise intensity, tune based on listening tests
                audio = audio + 0.005 * noise

        # Truncate long audio (limit to 30 seconds)
        if len(audio) > 16000 * 30:
            audio = audio[: 16000 * 30]

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        labels = self.processor.tokenizer(item["text"]).input_ids
        return {"input_features": input_features, "labels": labels}


def main():
    check_gpu()

    # Configure 4-bit Quantization (Drastically reduces VRAM usage)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading Model (4-bit Quantization)...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

    # Load Base Model
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    # Prepare LoRA Adapter Configuration
    # Preprocess model for k-bit training
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,  # LoRA Rank, larger means more parameters but better fitting
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # Fine-tune attention layers only
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    print("\nTrainable Parameters Stats:")
    model.print_trainable_parameters()

    # Enable augmentation for training set
    train_dataset = ATCInMemoryDataset(TRAIN_FILE, processor, augment=True)
    # No augmentation for test set
    test_dataset = ATCInMemoryDataset(TEST_FILE, processor, augment=False)

    # Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features):
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Set padding labels to -100 so loss ignores them
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,  # LoRA low VRAM allows larger batch
        gradient_accumulation_steps=2,
        learning_rate=1e-3,  # LoRA LR is typically larger than full fine-tuning (1e-3 vs 1e-5)
        warmup_steps=50,
        max_steps=1000,  # Train for 1000 steps, covering data multiple times
        fp16=True,  # Mixed precision training
        # Critical Fix: Use eval_strategy instead of evaluation_strategy
        eval_strategy="steps",
        predict_with_generate=True,  # Generate text during validation to calculate WER
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # Prevent custom columns in Dataset from being removed
        report_to=["tensorboard"],  # Logging report
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("\nStarting LoRA Fine-tuning...")
    trainer.train()

    print(f"\nTraining Complete! Saving LoRA Adapter...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
