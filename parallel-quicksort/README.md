# Parallel Quicksort - Performance Evaluation

This project demonstrates the performance benefits of parallel quicksort compared to sequential quicksort, with implementations in both C++ and Python.

## Project Overview

The project includes:
- Sequential Quicksort implementation
- Parallel Quicksort using multiple threads/processes
- Performance comparison with metrics (Speedup & Efficiency)
- Verification of sorting correctness

## Available Implementations

### 1. C++ Version (`parallel_quicksort.cpp`)
- Uses Windows threads (`<windows.h>`)
- Platform: Windows only
- High-performance implementation with low overhead

**Compilation:**
```bash
g++ parallel_quicksort.cpp -o parallel_quicksort.exe
```

**Run:**
```bash
./parallel_quicksort.exe
```

### 2. Python Version (`parallel_quicksort.py`)
- Uses `multiprocessing` and `concurrent.futures`
- Platform: Cross-platform (Windows, Linux, macOS)
- Easy to read and modify

**Run:**
```bash
python3 parallel_quicksort.py
```

## How It Works

### Algorithm
Both implementations use the quicksort algorithm with parallelization:

1. **Sequential Quicksort**: Standard recursive quicksort on a single thread/process
2. **Parallel Quicksort**:
   - Partitions the array recursively
   - Distributes sub-arrays across multiple threads/processes
   - Each thread/process sorts its portion independently
   - Results are combined in sorted order

### Parallelization Strategy
- Uses a depth parameter to control parallelism
- `depth=0`: 1 thread (sequential)
- `depth=1`: 2 threads
- `depth=2`: 4 threads
- `depth=3`: 8 threads

## Performance Metrics

The program measures and reports:

1. **Time**: Execution time for each configuration
2. **Speedup**: Sequential Time / Parallel Time
3. **Efficiency**: Speedup / Number of Threads
   - Ideal efficiency = 1.0
   - Decreases with more threads due to overhead

## Sample Output

```
==============================================================
  PARALLEL QUICKSORT - PERFORMANCE EVALUATION (Python)
==============================================================

  Array Size       : 1,000,000 elements
  Available CPUs   : 8
  Thread counts    : 1, 2, 4, 8
==============================================================

  Threads      Time (sec)       Speedup        Efficiency
  ------------ --------------- ------------- -------------
  1            1.3096           1.00           1.00
  2            0.8689           1.51           0.75
  4            0.7169           1.83           0.46
  8            0.6255           2.09           0.26

  Best Speedup : 2.09x with 8 threads
  Time Saved   : 0.6841 seconds (52.2%)
```

## Key Differences Between Implementations

| Feature | C++ Version | Python Version |
|---------|-------------|----------------|
| Platform | Windows only | Cross-platform |
| Threading | Windows threads | ProcessPoolExecutor |
| Overhead | Very low | Moderate (process creation) |
| Performance | Faster | Slightly slower |
| Readability | More complex | Easier to understand |
| Parallelism | True (native threads) | True (multiprocessing) |

## Notes

- Both versions use a fixed random seed (42) for reproducibility
- Array size: 1,000,000 elements by default
- The Python version may show lower speedup due to process creation overhead
- Efficiency naturally decreases with more threads due to:
  - Thread/process creation overhead
  - Synchronization costs
  - Load imbalancing
  - Memory bandwidth limitations

## LP6 - High Performance Computing

This project is part of the LP6 (Laboratory Practice 6) coursework focusing on High Performance Computing concepts and parallel algorithm implementation.
