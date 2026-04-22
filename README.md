# LP5 High Performance Computing & Deep Learning Labs

This repository contains two lab assignments for the LP5 coursework:
1. **Gender & Age Detection** (Deep Learning)
2. **Parallel Quicksort Performance Evaluation** (High Performance Computing)

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
