# LP5 High Performance Computing & Deep Learning Labs

This repository contains three lab assignments for the LP5 coursework:
1. **Gender & Age Detection** (Deep Learning)
2. **Parallel Quicksort Performance Evaluation** (High Performance Computing)
3. **Fine-Tuning Transformers for Sentiment Analysis** (NLP)

---

## How to Clone this Repository

To download this code to your local machine, run the following command in your terminal or command prompt:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```
*(Make sure to replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details).*

---

## Project 1: Gender & Age Detection

This project uses deep learning (TensorFlow/OpenCV) to detect gender and estimate age from live webcam feeds or images.

### Installation Steps (Libraries)
You need Python installed. Open your terminal, navigate to the `gender-age-detection` folder, and install the required libraries:

```bash
cd gender-age-detection
pip install -r requirements.txt
```

*(This will install `tensorflow`, `opencv-python`, and `numpy`)*

### How to Run
1. **Download the pre-trained models:**
   ```bash
   python download_models.py
   ```
2. **Run the live webcam detection:**
   ```bash
   python main.py
   ```
   *(Press 'q' to quit the webcam window).*

---

## Project 2: Parallel Quicksort Algorithm

This project evaluates the performance enhancement of the Quicksort algorithm using multi-threading (C++ Windows Threads). It compares sequential execution vs. parallel execution (2, 4, 8 threads, etc.).

### Installation Steps
Because this is written in pure C++ using standard Windows libraries, **no external installations are required** other than a basic C++ compiler like `g++` (MinGW).

### How to Run (Commands)
Navigate to the parallel quicksort folder:
```bash
cd parallel-quicksort
```

**Step 1: Compile the C++ code**
```bash
g++ parallel_quicksort.cpp -o parallel_quicksort.exe
```

**Step 2: Execute the program**
```bash
.\parallel_quicksort.exe
```
This will output a performance summary table showing Execution Time, Speedup, and Efficiency.

---

## Project 3: Fine-Tuning Transformers for Sentiment Analysis

### What It Does

This project **fine-tunes a pre-trained DistilBERT transformer model** to classify movie reviews as **Positive** or **Negative** (sentiment analysis). Here is what the script does step-by-step:

| Step | What Happens |
|------|-------------|
| **Step 1** | Downloads the **IMDB Movie Reviews** dataset (2000 train / 500 test samples) from Hugging Face |
| **Step 2** | **Tokenizes** the raw text into numerical tokens that the transformer can understand |
| **Step 3** | Loads the **pre-trained DistilBERT** model (~66 million parameters) and adds a classification head on top |
| **Step 4** | Sets up **Accuracy** and **F1 Score** evaluation metrics |
| **Step 5** | Configures training parameters (learning rate, batch size, epochs) |
| **Step 6** | **Fine-tunes** (trains) the model for 3 epochs — the model learns to distinguish positive vs negative reviews |
| **Step 7** | **Evaluates** the trained model on the test set and **saves** the fine-tuned model to disk |
| **Step 8** | Tests the model on **custom sentences** you can modify, showing real-time predictions with confidence scores |

### Key Concepts (For Viva)

- **Transfer Learning:** Instead of training a model from scratch, we take a model (DistilBERT) already trained on millions of text documents and adapt it for our specific task.
- **Fine-Tuning:** We update the pre-trained model's weights on our own dataset so it specializes in sentiment classification.
- **Tokenization:** Converting human-readable text into numerical IDs that the model processes (e.g., "hello" → [101, 7592, 102]).
- **DistilBERT:** A smaller, faster version of BERT that retains 97% of BERT's performance while being 60% faster.

### Installation Steps

Navigate to the project folder and install required libraries:

```bash
cd transformer-finetuning-project
pip install transformers datasets evaluate accelerate scikit-learn torch
```

### How to Run

```bash
cd transformer-finetuning-project
python run_finetuning.py
```

Training takes approximately **25 minutes on CPU** or **3 minutes on GPU**.

### Expected Output

```
============================================================
  FINAL EVALUATION RESULTS
============================================================
  Loss     : 0.4334
  Accuracy : 0.8240  (82.4%)
  F1 Score : 0.8255
============================================================

  Input : "This movie was an absolute waste of time. I hated it."
  Result: >> Negative  (Confidence: 0.9023)

  Input : "A brilliant masterpiece with phenomenal acting!"
  Result: >> Positive  (Confidence: 0.9133)

  Input : "I loved every single moment. A truly unforgettable experience!"
  Result: >> Positive  (Confidence: 0.8685)
============================================================
  PROJECT COMPLETE -- All steps finished successfully!
============================================================
```

### Project Files

| File | Description |
|------|-------------|
| `run_finetuning.py` | Main Python script — run this from the terminal |
| `Fine_Tuning_DistilBERT.ipynb` | Jupyter Notebook version (for Google Colab) |
| `Project_Report_README.md` | Detailed project report with explanations |
| `sentiment_model/` | Saved fine-tuned model (created after running) |

