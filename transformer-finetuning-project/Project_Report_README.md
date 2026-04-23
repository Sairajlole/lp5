# Mini Project: Fine-Tuning Transformers

## 1. Best Project Selection
**Recommended Task: Text Classification using DistilBERT**

**Why is it the best?**
* **Easy and Fast:** Text classification is straightforward. Using `DistilBERT` (a lighter, faster version of BERT) allows the model to train within minutes on a free Google Colab GPU.
* **Strong Results:** Even with a small subset of data, pre-trained models achieve high accuracy (85%+), which is excellent for a mini-project demonstration and presentations.
* **Good for Viva/Defense:** The architecture, tokenization process, and evaluation metrics (Accuracy, F1-Score) are simple to explain to examiners.
* **Free Tools:** Perfectly suited for Hugging Face libraries (`transformers`, `datasets`) and runs flawlessly on Google Colab's free tier.

## 2. Full Project Title
**"Sentiment Analysis of Movie Reviews via Fine-Tuning Pre-Trained Transformer Models"**

## 3. Dataset Recommendation
**Dataset:** `imdb` (Internet Movie Database Reviews) available directly via Hugging Face `datasets`.
* **Why chosen:** It's the most standard and widely understood benchmark for sentiment analysis. The labels are simple (Positive/Negative), making the evaluation intuitive.
* **Dataset Size & Labels:** The original dataset has 25,000 training and 25,000 testing rows. For this mini-project, we sample a smaller subset (2000 train, 500 test) to ensure the training completes in just a few minutes during a live demonstration.
* **Labels:** `0` (Negative), `1` (Positive).

## 4. Full Working Code
*The complete, runnable code is provided in the `Fine_Tuning_DistilBERT.ipynb` notebook inside this folder. You can upload this directly to Google Colab and run all cells.*

## 5. Step-by-Step Code Explanation

### Step 1: Install Dependencies
We install `transformers`, `datasets`, `evaluate`, and `accelerate`. These libraries provide the model architecture, the dataset API, the evaluation metrics, and the training loop optimization required for PyTorch.

### Step 2: Load and Prepare the Dataset
We load the `imdb` dataset. To make training fast for academic purposes, we shuffle and select a smaller subset (2000 for training, 500 for testing). 

### Step 3: Tokenization
Transformers cannot read raw text directly; they need numerical representations. We use `AutoTokenizer` to convert our text reviews into token IDs. We apply padding and truncation so all inputs have a uniform length of 128 tokens.

### Step 4: Load the Pre-trained Model
We load `distilbert-base-uncased` with a sequence classification head (`AutoModelForSequenceClassification`). We specify `num_labels=2` because our task is binary classification (Positive vs. Negative).

### Step 5: Define Evaluation Metrics
We load the accuracy and F1 score metrics using the `evaluate` library. The `compute_metrics` function will calculate these at the end of each training epoch to show how well the model is learning.

### Step 6: Fine-Tuning (Training)
We use the Hugging Face `Trainer` API. We define `TrainingArguments` like learning rate (`2e-5`), batch size, and number of epochs (`3`). The `Trainer` completely abstracts away the complex PyTorch training loops, making the code clean and readable.

### Step 7: Evaluate and Save
After training completes, we call `trainer.evaluate()` to see our final metrics on the test set. Finally, we save the fine-tuned model and tokenizer to a local directory so it can be reused later without retraining.

### Step 8: Test on Custom Input
We create a custom inference `pipeline` using our newly fine-tuned model and test it with custom sentences (e.g., *"This movie was absolutely fantastic!"*) to see the model's prediction in real-time.

## 6. Expected Output

**Training Logs (Approximate):**
```text
Epoch   Training Loss   Validation Loss   Accuracy   F1 Score
1       0.453200        0.312450          0.865      0.860
2       0.215400        0.285600          0.880      0.875
3       0.112300        0.301200          0.892      0.885
```

**Final Evaluation:**
```text
{'eval_loss': 0.3012, 'eval_accuracy': 0.892, 'eval_f1': 0.885, 'eval_runtime': 4.52}
```

**Custom Prediction Output:**
```text
Text: 'This movie was an absolute waste of time. I hated it.'
Prediction: Negative (Confidence: 0.9854)

Text: 'A brilliant masterpiece with phenomenal acting!'
Prediction: Positive (Confidence: 0.9912)
```
