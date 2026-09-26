# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/comparison_train_bert_srd.py
# What it does: Side experiment following a published sequential ASR -> speaker-role baseline: BERT tags each word as PILOT or ATCO. Written with heavy LLM assistance from a paper.
# Known problems:
#   - Turns with unknown speaker are labelled by a keyword heuristic, so BERT partly learns the heuristic.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# Configuration
MODEL_CHECKPOINT = "google-bert/bert-base-uncased"
OUTPUT_DIR = "comparison_bert_srd_model"

# Update paths to match your folder structure
TRAIN_JSON = "processed_data/ner_dataset_raw_split/train_raw.json"
TEST_JSON = "processed_data/ner_dataset_raw_split/test_raw.json"

# Hyperparameters from Paper Section 5 [cite: 121]
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WARMUP_STEPS = 25
PATIENCE = 5

# Label Definitions
LABEL_LIST = ["PILOT", "ATCO"]
label2id = {"PILOT": 0, "ATCO": 1}
id2label = {0: "PILOT", 1: "ATCO"}

# NEW: Heuristic Logic to fix "Unknown" labels on the fly
ATCO_KEYWORDS = [
    "WIND",
    "DEGREES",
    "KNOTS",
    "QNH",
    "SQUAWK",
    "IDENT",
    "RADAR",
    "SERVICE",
    "REPORT",
    "CLEARED",
    "CONTACT",
    "CROSS",
    "HOLD",
    "LINE UP",
    "PUSHBACK",
    "DESCEND",
    "CLIMB",
    "MAINTAIN",
    "TURN",
]


def guess_speaker_role(text):
    """
    Heuristically guesses if the speaker is ATCO or PILOT based on phraseology.
    This allows training to proceed even if labels are 'Unknown'.
    """
    text_upper = text.upper()

    # Check for Controller-specific keywords
    for word in ATCO_KEYWORDS:
        if word in text_upper:
            return "ATCO"

    # Length heuristic: ATCO instructions are often longer/complex
    if len(text_upper.split()) > 12:
        return "ATCO"

    # Default assumption: Pilots often give short readbacks
    return "PILOT"


def load_data_from_json(file_path):
    """
    Parses JSON to create word-level tags for BERT.
    Auto-fixes 'Unknown' labels in memory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tokens_list = []
    ner_tags_list = []

    found_count = 0
    fixed_count = 0

    print(f"Scanning {file_path}...")

    for session in raw_data:
        for turn in session.get("dialogue", []):
            text = turn.get("text", "").strip()

            # Get the raw label from JSON
            role = turn.get("speaker", turn.get("role", "Unknown")).upper()

            # FIX: If Unknown, guess it based on the text content
            if role == "UNKNOWN":
                role = guess_speaker_role(text)
                fixed_count += 1

            # Validation: Skip if text is empty
            words = text.split()
            if not words:
                continue

            # Map Role to ID
            # Logic: If it contains ATC-related terms, mark as ATCO ID
            if any(
                x in role for x in ["ATC", "TOWER", "GROUND", "CONTROLLER", "DEP", "APP", "ATCO"]
            ):
                label_id = label2id["ATCO"]
            else:
                label_id = label2id["PILOT"]

            # Assign the SAME label to every word in the turn [cite: 105]
            tokens_list.append(words)
            ner_tags_list.append([label_id] * len(words))
            found_count += 1

    print(f"Report: Loaded {found_count} turns. (Auto-labeled {fixed_count} 'Unknown' speakers)")

    if found_count == 0:
        raise ValueError("No valid data found even after auto-labeling!")

    return Dataset.from_dict({"tokens": tokens_list, "ner_tags": ner_tags_list})


def main():
    # Prepare Data
    print("Loading Data...")
    try:
        train_dataset = load_data_from_json(TRAIN_JSON)
        test_dataset = load_data_from_json(TEST_JSON)
    except Exception as e:
        print(e)
        return

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

    # Function to align labels with BERT sub-word tokens
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])  # First token of word gets label
                else:
                    label_ids.append(-100)  # Sub-tokens get ignored
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing...")
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

    # Model
    model = BertForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id
    )

    # Metrics
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=WARMUP_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    # Train
    print("Starting BERT Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"BERT SRD Model Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
