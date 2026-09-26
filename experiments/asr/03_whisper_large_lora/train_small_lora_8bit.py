# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/80%train.py
# What it does: Side branch: Whisper-small in 8-bit with LoRA (r=32), 3 epochs, best checkpoint by WER.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper_atco_lora_final"
TRAIN_DATA = "processed_data/train.jsonl"
TEST_DATA = "processed_data/test.jsonl"
LANGUAGE = "English"
TASK = "transcribe"

# VRAM Optimization
BATCH_SIZE = 8
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-3
EPOCHS = 3

# Path Fixer
FOUND_AUDIO_DIRS = set()


def scan_for_audio_files(base_dir):
    print(f'Scanning {base_dir} for audio files...')
    wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    dirs = set()
    for f in wav_files:
        dirs.add(os.path.dirname(f))
    print(f'Found {len(dirs)} directories containing audio.')
    return dirs


def fix_audio_path(batch):
    path = batch["audio_file"]
    if os.path.exists(path):
        return batch
    filename = os.path.basename(path)
    for d in FOUND_AUDIO_DIRS:
        possible_path = os.path.join(d, filename)
        if os.path.exists(possible_path):
            batch["audio_file"] = possible_path
            return batch
    return batch


# Data Prep
print("Loading Datasets...")
data_files = {"train": TRAIN_DATA, "test": TEST_DATA}
dataset = load_dataset("json", data_files=data_files)

FOUND_AUDIO_DIRS = scan_for_audio_files(os.getcwd())
print("Fixing Audio Paths...")
dataset = dataset.map(fix_audio_path)
dataset = dataset.filter(lambda x: os.path.exists(x["audio_file"]))
print(f"Dataset size: Train={len(dataset['train'])}, Test={len(dataset['test'])}")

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
normalizer = BasicTextNormalizer()

dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    audio = batch["audio_file"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["ASR_transcript_clean"]).input_ids
    return batch


print("Processing Features...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)


# Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Metrics
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]
    pred_str = [p if p.strip() else "" for p in pred_str]
    label_str = [l if l.strip() else "" for l in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Model and forward wrapper
print(f"Loading Model in 8-bit: {MODEL_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)

# Filter arguments before calling Whisper.
# Filter extra Trainer/PEFT arguments.
orig_forward = model.forward


def whisper_forward(input_features, *args, **kwargs):
    # Expanded blocklist of arguments to strictly ignore
    invalid_args = ["input_ids", "inputs_embeds", "num_items_in_batch"]
    for arg in invalid_args:
        if arg in kwargs:
            kwargs.pop(arg)
    return orig_forward(input_features, *args, **kwargs)


model.forward = whisper_forward
print("Applied the Whisper forward wrapper.")

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False

# Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    num_train_epochs=EPOCHS,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=128,
    save_total_limit=2,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to=["tensorboard"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Train
print("Starting QLoRA Training...")
trainer.train()

print("Saving LoRA Adapter...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
