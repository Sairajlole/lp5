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

### 2. Python Version - Multiprocessing (`parallel_quicksort.py`)
- Uses `multiprocessing` and `concurrent.futures`
- Platform: Cross-platform (Windows, Linux, macOS)
- Easy to read and modify
- No external dependencies

**Run:**
```bash
python3 parallel_quicksort.py
```

### 3. Python Version - MPI (`parallel_quicksort_mpi.py`)
- Uses `mpi4py` (Message Passing Interface)
- Platform: Cross-platform (Windows, Linux, macOS)
- Industry-standard parallel computing library
- Suitable for distributed computing and HPC clusters

**Requirements:**
```bash
# Install MPI (OpenMPI or MPICH)
# Ubuntu/Debian:
sudo apt-get install openmpi-bin libopenmpi-dev

# macOS:
brew install open-mpi

# Install mpi4py
pip install mpi4py
```

**Run:**
```bash
# Run with different number of processes
mpiexec -n 1 python parallel_quicksort_mpi.py  # Sequential
mpiexec -n 2 python parallel_quicksort_mpi.py  # 2 processes
mpiexec -n 4 python parallel_quicksort_mpi.py  # 4 processes
mpiexec -n 8 python parallel_quicksort_mpi.py  # 8 processes
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

**C++ and Python (Multiprocessing):**
- Uses a depth parameter to control parallelism
- `depth=0`: 1 thread (sequential)
- `depth=1`: 2 threads
- `depth=2`: 4 threads
- `depth=3`: 8 threads

**Python (MPI):**
- Uses number of MPI processes specified via `mpiexec -n <N>`
- Data is scattered across all processes
- Each process sorts its chunk independently
- Results are gathered and merged at root process
- Supports distributed memory parallelism

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

| Feature | C++ Version | Python (Multiprocessing) | Python (MPI) |
|---------|-------------|--------------------------|--------------|
| Platform | Windows only | Cross-platform | Cross-platform |
| Threading/Library | Windows threads | ProcessPoolExecutor | mpi4py (MPI) |
| Overhead | Very low | Moderate | Low-Moderate |
| Performance | Fastest | Good | Very Good |
| Readability | More complex | Easier | Moderate |
| Parallelism | True (threads) | True (processes) | True (MPI processes) |
| Dependencies | None (Windows API) | None (stdlib) | mpi4py, MPI runtime |
| Use Case | Windows HPC | General purpose | HPC clusters |
| Scalability | Limited | Multi-core | Multi-node capable |

## Notes

- All versions use a fixed random seed (42) for reproducibility
- Array size: 1,000,000 elements by default
- The Python versions may show lower speedup due to process creation overhead
- Efficiency naturally decreases with more threads/processes due to:
  - Thread/process creation overhead
  - Synchronization costs
  - Load imbalancing
  - Memory bandwidth limitations
  - Communication overhead (especially in MPI)

### MPI-Specific Notes
- The MPI version is ideal for HPC environments and can scale across multiple nodes
- Use `mpiexec` or `mpirun` to launch MPI programs
- The number of processes should ideally match or be less than available CPU cores
- For distributed systems, ensure MPI is properly configured across all nodes
- The MPI version uses scatter/gather collective operations for data distribution

## LP6 - High Performance Computing

This project is part of the LP6 (Laboratory Practice 6) coursework focusing on High Performance Computing concepts and parallel algorithm implementation.
