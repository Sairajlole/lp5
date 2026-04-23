"""
=============================================================================
  Sentiment Analysis via Fine-Tuning DistilBERT
  Mini Project: Fine-Tuning Transformers
=============================================================================
  This script fine-tunes a pre-trained DistilBERT model for binary
  text classification (sentiment analysis) on the IMDb movie reviews dataset.
  
  Works on both CPU and GPU. On CPU with the small subset, training takes
  approximately 10-15 minutes.
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# --- Configuration -----------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 128
TRAIN_SIZE = 500        # Subset size for fast training
TEST_SIZE = 200         # Subset size for evaluation
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SAVE_DIR = "./sentiment_model"
OUTPUT_DIR = "./results"

# --- Step 0: Device Setup ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print("  Fine-Tuning DistilBERT for Sentiment Analysis")
print("=" * 60)
print(f"  Device     : {device.upper()}")
print(f"  Model      : {MODEL_NAME}")
print(f"  Train size : {TRAIN_SIZE}")
print(f"  Test size  : {TEST_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print("=" * 60)

# --- Step 1: Load Dataset ----------------------------------------------------
print("\n[Step 1/8] Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_dataset = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
test_dataset = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))

print(f"  [OK] Train samples: {len(train_dataset)}")
print(f"  [OK] Test samples : {len(test_dataset)}")
print(f"  [OK] Labels       : 0 = Negative, 1 = Positive")

# --- Step 2: Tokenization ----------------------------------------------------
print("\n[Step 2/8] Tokenizing datasets...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
print("  [OK] Tokenization complete")

# --- Step 3: Load Pre-trained Model ------------------------------------------
print("\n[Step 3/8] Loading pre-trained DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  [OK] Total parameters     : {total_params:,}")
print(f"  [OK] Trainable parameters : {trainable_params:,}")

# --- Step 4: Define Evaluation Metrics ----------------------------------------
print("\n[Step 4/8] Setting up evaluation metrics...")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

print("  [OK] Accuracy and F1 metrics loaded")

# --- Step 5: Training Arguments -----------------------------------------------
print("\n[Step 5/8] Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=25,
    report_to="none",
    use_cpu=(device == "cpu"),
)
print("  [OK] Training arguments configured")

# --- Step 6: Fine-Tune (Train) ------------------------------------------------
print("\n[Step 6/8] Starting fine-tuning...")
print("-" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("-" * 60)
print("  [OK] Training complete!")

# --- Step 7: Evaluate & Save --------------------------------------------------
print("\n[Step 7/8] Evaluating on test set...")
results = trainer.evaluate()

print("\n" + "=" * 60)
print("  FINAL EVALUATION RESULTS")
print("=" * 60)
print(f"  Loss     : {results['eval_loss']:.4f}")
print(f"  Accuracy : {results['eval_accuracy']:.4f}  ({results['eval_accuracy']*100:.1f}%)")
print(f"  F1 Score : {results['eval_f1']:.4f}")
print("=" * 60)

print(f"\n  Saving model to '{SAVE_DIR}'...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("  [OK] Model and tokenizer saved!")

# --- Step 8: Test on Custom Input ---------------------------------------------
print("\n[Step 8/8] Testing on custom inputs...")
print("-" * 60)

sentiment_pipeline = pipeline(
    "text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    device=0 if device == "cuda" else -1,
)

custom_texts = [
    "This movie was an absolute waste of time. I hated it.",
    "A brilliant masterpiece with phenomenal acting!",
    "It was okay, but the ending could have been better.",
    "One of the worst films I have ever watched. Terrible script.",
    "I loved every single moment. A truly unforgettable experience!",
]

for text in custom_texts:
    prediction = sentiment_pipeline(text)
    label = ">> Positive" if prediction[0]["label"] == "LABEL_1" else ">> Negative"
    score = prediction[0]["score"]
    print(f"\n  Input : \"{text}\"")
    print(f"  Result: {label}  (Confidence: {score:.4f})")

print("\n" + "=" * 60)
print("  PROJECT COMPLETE -- All steps finished successfully!")
print("=" * 60)
