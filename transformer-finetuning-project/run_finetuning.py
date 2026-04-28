"""
Fine-Tuning DistilBERT for Sentiment Analysis (IMDb)
"""

import warnings
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate

warnings.filterwarnings("ignore")

# --- Config ------------------------------------------------------------------
MODEL_NAME  = "distilbert-base-uncased"
SAVE_DIR    = "./sentiment_model"
TRAIN_SIZE  = 500
TEST_SIZE   = 200
MAX_LENGTH  = 128
EPOCHS      = 3
BATCH_SIZE  = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()} | Model: {MODEL_NAME} | Train: {TRAIN_SIZE} | Epochs: {EPOCHS}")

# --- Dataset -----------------------------------------------------------------
print("\nLoading IMDb dataset...")
ds = load_dataset("imdb")
train_ds = ds["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
test_ds  = ds["test"].shuffle(seed=42).select(range(TEST_SIZE))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_tok = train_ds.map(tokenize, batched=True)
test_tok  = test_ds.map(tokenize, batched=True)
print(f"Tokenized — train: {len(train_tok)}, test: {len(test_tok)}")

# --- Model -------------------------------------------------------------------
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# --- Metrics -----------------------------------------------------------------
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1_metric.compute(predictions=preds,  references=labels)["f1"],
    }

# --- Training ----------------------------------------------------------------
print("\nFine-tuning...")
args = TrainingArguments(
    output_dir       = "./results",
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate    = 2e-5,
    weight_decay     = 0.01,
    eval_strategy    = "epoch",
    save_strategy    = "epoch",
    load_best_model_at_end = True,
    logging_steps    = 25,
    report_to        = "none",
    use_cpu          = (device == "cpu"),
)

trainer = Trainer(
    model            = model,
    args             = args,
    train_dataset    = train_tok,
    eval_dataset     = test_tok,
    processing_class = tokenizer,
    compute_metrics  = compute_metrics,
)

trainer.train()

# --- Evaluate & Save ---------------------------------------------------------
results = trainer.evaluate()
print(f"\nResults — Loss: {results['eval_loss']:.4f} | Accuracy: {results['eval_accuracy']*100:.1f}% | F1: {results['eval_f1']:.4f}")

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to '{SAVE_DIR}'")

# --- Inference Test ----------------------------------------------------------
print("\nTesting on custom inputs...")
pipe = pipeline("text-classification", model=SAVE_DIR, tokenizer=SAVE_DIR, device=0 if device == "cuda" else -1)

samples = [
    "This movie was an absolute waste of time. I hated it.",
    "A brilliant masterpiece with phenomenal acting!",
    "It was okay, but the ending could have been better.",
    "One of the worst films I have ever watched. Terrible script.",
    "I loved every single moment. A truly unforgettable experience!",
]

for text in samples:
    pred = pipe(text)[0]
    label = "Positive" if pred["label"] == "LABEL_1" else "Negative"
    print(f"  [{label} {pred['score']:.2f}] {text}")

print("\nDone!")
