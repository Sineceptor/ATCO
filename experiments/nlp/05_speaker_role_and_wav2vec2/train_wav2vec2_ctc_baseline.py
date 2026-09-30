# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/comparison_train_pure_asr.py
# What it does: wav2vec2-base-960h CTC fine-tune as a non-Whisper ASR baseline (lr 4e-4, 2000 steps).
# Source: the ASR settings in section 5 of Blatt, A., Krishnan, A. and Klakow, D. Joint vs Sequential Speaker-Role
#   Detection and Automatic Speech Recognition for Air-traffic Control. Interspeech 2024 (2000 steps, learning rate 4e-4).
# Known problems:
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate

# Configuration
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = "comparison_pure_asr_model"

# Update these to match your exact paths
TRAIN_JSON = "processed_data/ner_dataset_raw_split/train_raw.json"
TEST_JSON = "processed_data/ner_dataset_raw_split/test_raw.json"
AUDIO_BASE_DIR = "processed_data/audio"  # Path to the folder containing .wav files

# Hyperparameters
LEARNING_RATE = 4e-4
WARMUP_STEPS = 1000
MAX_STEPS = 2000
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8


def load_data_from_json(json_path):
    """
    Parses JSON where audio filenames are nested inside the 'dialogue' list.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    paths = []
    sentences = []

    # Debug counters
    missing_files = 0
    found_files = 0

    print(f"Scanning {json_path}...")

    for session in data:
        # Loop through the dialogue turns FIRST
        for turn in session.get("dialogue", []):
            # Get audio filename from the turn object
            audio_filename = turn.get("audio", "")

            if not audio_filename:
                continue

            # Construct absolute path
            full_audio_path = os.path.join(AUDIO_BASE_DIR, audio_filename)

            # Check if file exists
            if not os.path.exists(full_audio_path):
                missing_files += 1
                if missing_files == 1:
                    print(f"ERROR: Cannot find audio file at: {full_audio_path}")
                continue

            # Get text
            text = turn.get("text", "").strip().upper()

            # Validation
            if len(text) < 2:
                continue

            found_files += 1
            paths.append(full_audio_path)
            sentences.append(text)

    print(f'Report: Found {found_files} valid audio-text pairs. Missing {missing_files} files.')

    if len(paths) == 0:
        raise ValueError("No valid training data found! Please fix AUDIO_BASE_DIR.")

    return Dataset.from_dict({"path": paths, "sentence": sentences})


def main():
    # Load Datasets
    print(f"Loading data from JSON...")
    try:
        train_dataset = load_data_from_json(TRAIN_JSON)
        test_dataset = load_data_from_json(TEST_JSON)
    except Exception as e:
        print(e)
        return

    print(f'Loaded {len(train_dataset)} training samples.')

    # Setup Processor
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_CHECKPOINT)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Audio Preprocessing
    train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("path", Audio(sampling_rate=16000))

    # FIXED PREPARE FUNCTION
    def prepare_dataset(batch):
        audio = batch["path"]

        # Process Audio (Input)
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]

        # Process Text (Labels) - Fixed for new transformers version
        # We explicitly use text=... instead of the deprecated context manager
        batch["labels"] = processor(text=batch["sentence"]).input_ids

        return batch

    print("Preprocessing audio and text...")
    # num_proc=1 is safer on Windows to avoid process crashing
    train_dataset = train_dataset.map(
        prepare_dataset, remove_columns=["path", "sentence"], num_proc=1
    )
    test_dataset = test_dataset.map(
        prepare_dataset, remove_columns=["path", "sentence"], num_proc=1
    )

    # Data Collator
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Load Model
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_CHECKPOINT,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Trainer Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        eval_strategy="steps",  # Fixed argument name
        num_train_epochs=10,
        max_steps=MAX_STEPS,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    # Train
    print("Starting Pure ASR Training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Pure ASR Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
