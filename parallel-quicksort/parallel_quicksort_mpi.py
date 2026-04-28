"""
============================================================
 Performance Evaluation of Parallel Quicksort Algorithm
 Using Python with MPI (Message Passing Interface)

 LP6 - High Performance Computing
============================================================

 This program demonstrates:
   1. Sequential Quicksort on a large array
   2. Parallel Quicksort using MPI processes
   3. Performance comparison: Speedup & Efficiency

 Run:
   mpiexec -n <num_processes> python parallel_quicksort_mpi.py

 Examples:
   mpiexec -n 1 python parallel_quicksort_mpi.py  # Sequential
   mpiexec -n 2 python parallel_quicksort_mpi.py  # 2 processes
   mpiexec -n 4 python parallel_quicksort_mpi.py  # 4 processes
   mpiexec -n 8 python parallel_quicksort_mpi.py  # 8 processes
============================================================
"""

import time
import random
from mpi4py import MPI
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
# 3. PARALLEL QUICKSORT USING MPI
# ============================================================
def parallel_quicksort_mpi(comm, data: List[int]) -> List[int]:
    """
    Parallel quicksort using MPI.
    Each process sorts its portion, then results are merged.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        # Sequential case
        if data is not None and len(data) > 1:
            arr = data.copy()
            sequential_quicksort(arr, 0, len(arr) - 1)
            return arr
        return data if data is not None else []

    # Root process distributes data
    if rank == 0:
        # Partition data into chunks for each process
        chunk_size = len(data) // size
        chunks = []
        for i in range(size):
            start = i * chunk_size
            if i == size - 1:
                # Last chunk gets remaining elements
                end = len(data)
            else:
                end = (i + 1) * chunk_size
            chunks.append(data[start:end])
    else:
        chunks = None

    # Scatter chunks to all processes
    local_data = comm.scatter(chunks, root=0)

    # Each process sorts its local chunk
    if len(local_data) > 1:
        sequential_quicksort(local_data, 0, len(local_data) - 1)

    # Gather sorted chunks back to root
    sorted_chunks = comm.gather(local_data, root=0)

    # Root process merges sorted chunks
    if rank == 0:
        # Merge all sorted chunks using k-way merge
        result = merge_sorted_chunks(sorted_chunks)
        return result
    else:
        return None


def merge_sorted_chunks(chunks: List[List[int]]) -> List[int]:
    """
    Merge multiple sorted chunks into a single sorted array.
    Uses a simple approach of concatenating and sorting.
    For better performance, could use a heap-based k-way merge.
    """
    result = []
    for chunk in chunks:
        result.extend(chunk)

    # Since chunks are sorted but need to be merged properly
    # We use a simple merge approach
    if len(result) > 1:
        sequential_quicksort(result, 0, len(result) - 1)

    return result


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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ARRAY_SIZE = 1_000_000  # 1 million elements

    # Only root process prints headers and generates data
    if rank == 0:
        print_line()
        print("  PARALLEL QUICKSORT - PERFORMANCE EVALUATION (MPI)")
        print_line()
        print()
        print(f"  Array Size       : {ARRAY_SIZE:,} elements")
        print(f"  MPI Processes    : {size}")
        print_line()

        # Generate random data with fixed seed for reproducibility
        random.seed(42)
        original_data = [random.randint(0, 999999) for _ in range(ARRAY_SIZE)]
        print("\n  Random array generated successfully.\n")
    else:
        original_data = None

    # Synchronize all processes
    comm.Barrier()

    # ----------------------------------------------------------
    # Run MPI Parallel Sort
    # ----------------------------------------------------------
    if rank == 0:
        # Make a copy for parallel sort
        data = original_data.copy()
        print(f"  [MPI PARALLEL] Sorting with {size} process{'es' if size > 1 else ''}...")

    else:
        data = None

    # Time the parallel sort
    start_time = MPI.Wtime()

    sorted_data = parallel_quicksort_mpi(comm, data)

    end_time = MPI.Wtime()
    parallel_time = end_time - start_time

    # Only root process verifies and prints results
    if rank == 0:
        # Verify correctness
        if is_sorted(sorted_data):
            print("             [OK] Verification PASSED - Array is correctly sorted")
        else:
            print("             [FAIL] Verification FAILED - Array is NOT sorted!")

        print(f"             Time taken: {parallel_time:.4f} seconds")

        # ----------------------------------------------------------
        # Run Sequential Sort for comparison (only if size > 1)
        # ----------------------------------------------------------
        if size > 1:
            print(f"\n  [SEQUENTIAL] Sorting with 1 process for comparison...")

            data_seq = original_data.copy()
            start_seq = time.perf_counter()
            sequential_quicksort(data_seq, 0, len(data_seq) - 1)
            end_seq = time.perf_counter()
            sequential_time = end_seq - start_seq

            if is_sorted(data_seq):
                print("             [OK] Verification PASSED - Array is correctly sorted")
            else:
                print("             [FAIL] Verification FAILED - Array is NOT sorted!")

            print(f"             Time taken: {sequential_time:.4f} seconds")

            # ----------------------------------------------------------
            # Print Performance Summary
            # ----------------------------------------------------------
            print()
            print_line()
            print("  PERFORMANCE SUMMARY")
            print_line()
            print()
            print(f"  {'Configuration':<20} {'Time (sec)':<16} {'Speedup':<14} {'Efficiency':<14}")
            print(f"  {'-' * 20} {'-' * 15} {'-' * 13} {'-' * 13}")

            # Sequential results
            print(f"  {'Sequential (1)':<20} {sequential_time:<16.4f} {1.00:<14.2f} {1.00:<14.2f}")

            # Parallel results
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / size
            print(f"  {'Parallel (' + str(size) + ')':<20} {parallel_time:<16.4f} {speedup:<14.2f} {efficiency:<14.2f}")

            # ----------------------------------------------------------
            # Print Analysis
            # ----------------------------------------------------------
            print()
            print_line()
            print("  ANALYSIS")
            print_line()
            print()
            print(f"  Speedup      : {speedup:.2f}x with {size} processes")
            print(f"  Efficiency   : {efficiency:.2f} ({efficiency*100:.1f}%)")
            print(f"  Sequential   : {sequential_time:.4f} seconds")
            print(f"  Parallel     : {parallel_time:.4f} seconds")

            time_saved = sequential_time - parallel_time
            percent_saved = (time_saved / sequential_time) * 100.0 if sequential_time > 0 else 0
            print(f"  Time Saved   : {time_saved:.4f} seconds ({percent_saved:.1f}%)")

            print("\n  Key Observations:")
            print("  - Sequential sort uses standard Quicksort (single process)")
            print("  - Parallel sort distributes data across MPI processes")
            print("  - Each process sorts its chunk independently")
            print("  - Speedup = Sequential Time / Parallel Time")
            print("  - Efficiency = Speedup / Number of Processes (ideal = 1.0)")
            print("  - Communication overhead reduces efficiency")
        else:
            print("\n  Note: Running with 1 process (sequential mode)")
            print("  Run with: mpiexec -n <N> python parallel_quicksort_mpi.py")
            print("  where N > 1 for parallel comparison")

        print()
        print_line()
        print("  EXPERIMENT COMPLETE")
        print_line()
        print()


if __name__ == "__main__":
    main()
