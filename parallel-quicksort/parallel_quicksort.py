"""
============================================================
 Performance Evaluation of Parallel Quicksort Algorithm
 Using Python with Multiprocessing

 LP6 - High Performance Computing
============================================================

 This program demonstrates:
   1. Sequential Quicksort on a large array
   2. Parallel Quicksort using multiple processes
   3. Performance comparison: Speedup & Efficiency

 Run:
   python parallel_quicksort.py
============================================================
"""

import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List


# ============================================================
# 1. QUICKSORT - PARTITION FUNCTION
# ============================================================
def partition(arr: List[int], low: int, high: int) -> int:
    """
    Partitions the array around a pivot element.
    Returns the final position of the pivot.
    """
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# ============================================================
# 2. SEQUENTIAL QUICKSORT (standard recursive)
# ============================================================
def sequential_quicksort(arr: List[int], low: int, high: int) -> None:
    """
    Standard recursive quicksort - sorts in-place.
    """
    if low < high:
        pi = partition(arr, low, high)
        sequential_quicksort(arr, low, pi - 1)
        sequential_quicksort(arr, pi + 1, high)


# ============================================================
# 3. PARALLEL QUICKSORT
# ============================================================
def _sequential_sort_wrapper(arr: List[int]) -> List[int]:
    """
    Wrapper for sequential sort - used by parallel workers.
    """
    if len(arr) <= 1:
        return arr
    arr_copy = arr.copy()
    sequential_quicksort(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy


def _partition_array(arr: List[int], num_chunks: int) -> List[List[int]]:
    """
    Recursively partition array using quicksort partitioning.
    Returns list of chunks to be sorted independently.
    """
    if num_chunks <= 1 or len(arr) <= 1:
        return [arr]

    # Partition using last element as pivot
    pivot = arr[-1]
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    middle = [pivot]

    # Recursively partition left and right
    left_chunks = num_chunks // 2
    right_chunks = num_chunks - left_chunks

    left_partitions = _partition_array(left, left_chunks) if left else []
    right_partitions = _partition_array(right, right_chunks) if right else []

    # Return all partitions in order (they'll be sorted then concatenated)
    return left_partitions + [middle] + right_partitions


def parallel_quicksort(arr: List[int], depth: int) -> None:
    """
    Parallel quicksort that partitions array into chunks and sorts them in parallel.
    depth controls the level of parallelism (depth=n means up to 2^n processes).
    """
    if len(arr) <= 1:
        return

    num_processes = 2 ** depth if depth > 0 else 1

    if num_processes == 1:
        # Sequential sort
        sequential_quicksort(arr, 0, len(arr) - 1)
    else:
        # Partition into chunks
        chunks = _partition_array(arr, num_processes)

        # Sort chunks in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            sorted_chunks = list(executor.map(_sequential_sort_wrapper, chunks))

        # Concatenate results
        result = []
        for chunk in sorted_chunks:
            result.extend(chunk)

        # Update array in-place
        arr[:] = result


# ============================================================
# 4. VERIFY SORTED
# ============================================================
def is_sorted(arr: List[int]) -> bool:
    """
    Checks if the array is sorted in ascending order.
    """
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            return False
    return True


# ============================================================
# 5. PRINT SEPARATOR
# ============================================================
def print_line():
    """Prints a separator line."""
    print("=" * 62)


# ============================================================
# 6. MAIN - PERFORMANCE EVALUATION
# ============================================================
def main():
    ARRAY_SIZE = 1_000_000  # 1 million elements

    # Get number of available CPUs
    max_threads = mp.cpu_count()

    print_line()
    print("  PARALLEL QUICKSORT - PERFORMANCE EVALUATION (Python)")
    print_line()
    print()
    print(f"  Array Size       : {ARRAY_SIZE:,} elements")
    print(f"  Available CPUs   : {max_threads}")
    print(f"  Thread counts    : 1, 2, 4", end="")
    if max_threads > 4:
        print(f", {min(8, max_threads)}", end="")
    print()
    print_line()

    # Generate random data with fixed seed for reproducibility
    random.seed(42)
    original_data = [random.randint(0, 999999) for _ in range(ARRAY_SIZE)]
    print("\n  Random array generated successfully.\n")

    # Thread counts to test (depth = log2(threads))
    # depth=0 -> 1 thread, depth=1 -> 2 threads, depth=2 -> 4 threads, etc.
    test_cases = [
        {"label": 1, "depth": 0},  # Sequential (depth 0 = no parallel tasks)
        {"label": 2, "depth": 1},  # 2 threads  (depth 1 = 2 parallel branches)
        {"label": 4, "depth": 2},  # 4 threads  (depth 2 = 4 parallel branches)
    ]
    if max_threads >= 8:
        # depth 3 = up to 8 threads
        test_cases.append({"label": 8, "depth": 3})

    # Store results
    results = []
    sequential_time = 0.0

    # ----------------------------------------------------------
    # Run experiments for each thread count
    # ----------------------------------------------------------
    for tc in test_cases:
        # Make a fresh copy of the original data for each run
        data = original_data.copy()

        label = tc["label"]
        depth = tc["depth"]

        mode = "SEQUENTIAL" if label == 1 else "PARALLEL  "
        thread_word = "thread " if label == 1 else "threads"
        print(f"\n  [{mode}] Sorting with {label} {thread_word}...")

        # Time the sort
        start = time.perf_counter()

        if label == 1:
            # Pure sequential quicksort
            sequential_quicksort(data, 0, ARRAY_SIZE - 1)
        else:
            # Parallel quicksort
            parallel_quicksort(data, depth)

        end = time.perf_counter()
        elapsed = end - start

        # Store sequential time as baseline
        if label == 1:
            sequential_time = elapsed

        # Verify correctness
        if is_sorted(data):
            print("             [OK] Verification PASSED - Array is correctly sorted")
        else:
            print("             [FAIL] Verification FAILED - Array is NOT sorted!")

        print(f"             Time taken: {elapsed:.4f} seconds")

        # Calculate metrics
        speedup = sequential_time / elapsed if elapsed > 0 else 0
        efficiency = speedup / label

        results.append({
            "threads": label,
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency
        })

    # ----------------------------------------------------------
    # Print Performance Summary Table
    # ----------------------------------------------------------
    print()
    print_line()
    print("  PERFORMANCE SUMMARY")
    print_line()
    print()
    print(f"  {'Threads':<12} {'Time (sec)':<16} {'Speedup':<14} {'Efficiency':<14}")
    print(f"  {'-' * 12} {'-' * 15} {'-' * 13} {'-' * 13}")

    for r in results:
        print(f"  {r['threads']:<12} {r['time']:<16.4f} {r['speedup']:<14.2f} {r['efficiency']:<14.2f}")

    # ----------------------------------------------------------
    # Print Analysis
    # ----------------------------------------------------------
    best = max(results, key=lambda x: x['speedup'])

    print()
    print_line()
    print("  ANALYSIS")
    print_line()
    print()
    print(f"  Best Speedup : {best['speedup']:.2f}x with {best['threads']} threads")
    print(f"  Sequential   : {sequential_time:.4f} seconds")
    print(f"  Best Parallel: {best['time']:.4f} seconds")

    time_saved = sequential_time - best['time']
    percent_saved = (time_saved / sequential_time) * 100.0 if sequential_time > 0 else 0
    print(f"  Time Saved   : {time_saved:.4f} seconds ({percent_saved:.1f}%)")

    print("\n  Key Observations:")
    print("  - Sequential sort uses standard Quicksort (single process)")
    print("  - Parallel sort creates processes to sort sub-arrays concurrently")
    print("  - Speedup = Sequential Time / Parallel Time")
    print("  - Efficiency = Speedup / Number of Threads (ideal = 1.0)")
    print("  - Process creation overhead reduces efficiency at high thread counts")
    print()
    print_line()
    print("  EXPERIMENT COMPLETE")
    print_line()
    print()


if __name__ == "__main__":
    main()
