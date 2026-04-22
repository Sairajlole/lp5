/*
 * ============================================================
 *  Performance Evaluation of Parallel Quicksort Algorithm
 *  Using C++ with Windows Threads
 *
 *  LP5 - High Performance Computing
 * ============================================================
 *
 *  This program demonstrates:
 *    1. Sequential Quicksort on a large array
 *    2. Parallel Quicksort using multiple threads
 *    3. Performance comparison: Speedup & Efficiency
 *
 *  Compilation:
 *    g++ parallel_quicksort.cpp -o parallel_quicksort.exe
 *
 *  Run:
 *    parallel_quicksort.exe
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <windows.h>

using namespace std;

// ============================================================
// HIGH RESOLUTION TIMER (Windows)
// ============================================================
double getTime() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

// ============================================================
// 1. QUICKSORT - PARTITION FUNCTION
// ============================================================
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// ============================================================
// 2. SEQUENTIAL QUICKSORT (standard recursive)
// ============================================================
void sequentialQuickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        sequentialQuickSort(arr, low, pi - 1);
        sequentialQuickSort(arr, pi + 1, high);
    }
}

// ============================================================
// 3. PARALLEL QUICKSORT STRUCTURES
// ============================================================

struct ThreadArg {
    vector<int>* arr;
    int low;
    int high;
    int depth;
};

// Forward declaration
DWORD WINAPI parallelQuickSortThread(LPVOID param);

void parallelQuickSort(vector<int>& arr, int low, int high, int depth) {
    if (low >= high) return;

    int pi = partition(arr, low, high);

    if (depth > 0) {
        // Create thread for LEFT half
        ThreadArg leftArg = {&arr, low, pi - 1, depth - 1};
        HANDLE leftThread = CreateThread(
            NULL, 0, parallelQuickSortThread, &leftArg, 0, NULL
        );

        // Sort RIGHT half in current thread
        parallelQuickSort(arr, pi + 1, high, depth - 1);

        // Wait for left thread to finish
        WaitForSingleObject(leftThread, INFINITE);
        CloseHandle(leftThread);
    } else {
        // Fall back to sequential for deep recursion
        sequentialQuickSort(arr, low, pi - 1);
        sequentialQuickSort(arr, pi + 1, high);
    }
}

DWORD WINAPI parallelQuickSortThread(LPVOID param) {
    ThreadArg* arg = (ThreadArg*)param;
    parallelQuickSort(*(arg->arr), arg->low, arg->high, arg->depth);
    return 0;
}

// ============================================================
// 4. VERIFY SORTED
// ============================================================
bool isSorted(const vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

// ============================================================
// 5. PRINT SEPARATOR
// ============================================================
void printLine() {
    cout << string(62, '=') << endl;
}

// ============================================================
// 6. MAIN - PERFORMANCE EVALUATION
// ============================================================
int main() {
    const int ARRAY_SIZE = 1000000;  // 1 million elements

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    int maxThreads = sysInfo.dwNumberOfProcessors;

    printLine();
    cout << "  PARALLEL QUICKSORT - PERFORMANCE EVALUATION (C++)" << endl;
    printLine();
    cout << endl;
    cout << "  Array Size       : " << ARRAY_SIZE << " elements" << endl;
    cout << "  Available CPUs   : " << maxThreads << endl;
    cout << "  Thread counts    : 1, 2, 4";
    if (maxThreads > 4) cout << ", " << maxThreads;
    cout << endl;
    printLine();

    // Generate random data with fixed seed for reproducibility
    srand(42);
    vector<int> originalData(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        originalData[i] = rand() % 1000000;
    }
    cout << "\n  Random array generated successfully.\n";

    // Thread counts to test (depth = log2(threads))
    // depth=0 -> 1 thread, depth=1 -> 2 threads, depth=2 -> 4 threads, etc.
    struct TestCase {
        int label;   // number of threads (for display)
        int depth;   // recursion depth for parallelism
    };

    vector<TestCase> testCases = {
        {1, 0},  // Sequential (depth 0 = no parallel tasks)
        {2, 1},  // 2 threads  (depth 1 = 2 parallel branches)
        {4, 2},  // 4 threads  (depth 2 = 4 parallel branches)
    };
    if (maxThreads >= 8) {
        // depth 3 = up to 8 threads
        testCases.push_back({8, 3});
    }

    // Store results
    struct Result {
        int threads;
        double time;
        double speedup;
        double efficiency;
    };
    vector<Result> results;
    double sequentialTime = 0.0;

    // ----------------------------------------------------------
    // Run experiments for each thread count
    // ----------------------------------------------------------
    for (const auto& tc : testCases) {
        // Make a fresh copy of the original data for each run
        vector<int> data = originalData;

        cout << "\n  [" << (tc.label == 1 ? "SEQUENTIAL" : "PARALLEL  ")
             << "] Sorting with " << tc.label
             << (tc.label == 1 ? " thread " : " threads")
             << "..." << endl;

        // Time the sort
        double start = getTime();

        if (tc.label == 1) {
            // Pure sequential quicksort
            sequentialQuickSort(data, 0, ARRAY_SIZE - 1);
        } else {
            // Parallel quicksort
            parallelQuickSort(data, 0, ARRAY_SIZE - 1, tc.depth);
        }

        double end = getTime();
        double elapsed = end - start;

        // Store sequential time as baseline
        if (tc.label == 1) {
            sequentialTime = elapsed;
        }

        // Verify correctness
        if (isSorted(data)) {
            cout << "             [OK] Verification PASSED - Array is correctly sorted" << endl;
        } else {
            cout << "             [FAIL] Verification FAILED - Array is NOT sorted!" << endl;
        }

        cout << "             Time taken: " << fixed << setprecision(4) << elapsed << " seconds" << endl;

        // Calculate metrics
        double speedup = (elapsed > 0) ? sequentialTime / elapsed : 0;
        double efficiency = speedup / tc.label;

        results.push_back({tc.label, elapsed, speedup, efficiency});
    }

    // ----------------------------------------------------------
    // Print Performance Summary Table
    // ----------------------------------------------------------
    cout << endl;
    printLine();
    cout << "  PERFORMANCE SUMMARY" << endl;
    printLine();
    cout << endl;
    cout << "  " << left << setw(12) << "Threads"
         << setw(16) << "Time (sec)"
         << setw(14) << "Speedup"
         << setw(14) << "Efficiency" << endl;
    cout << "  " << string(12, '-') << " "
         << string(15, '-') << " "
         << string(13, '-') << " "
         << string(13, '-') << endl;

    for (const auto& r : results) {
        cout << "  " << left << setw(12) << r.threads
             << setw(16) << fixed << setprecision(4) << r.time
             << setw(14) << fixed << setprecision(2) << r.speedup
             << setw(14) << fixed << setprecision(2) << r.efficiency << endl;
    }

    // ----------------------------------------------------------
    // Print Analysis
    // ----------------------------------------------------------
    Result best = results[0];
    for (const auto& r : results) {
        if (r.speedup > best.speedup) best = r;
    }

    cout << endl;
    printLine();
    cout << "  ANALYSIS" << endl;
    printLine();
    cout << endl;
    cout << "  Best Speedup : " << fixed << setprecision(2) << best.speedup
         << "x with " << best.threads << " threads" << endl;
    cout << "  Sequential   : " << fixed << setprecision(4) << sequentialTime << " seconds" << endl;
    cout << "  Best Parallel: " << fixed << setprecision(4) << best.time << " seconds" << endl;

    double timeSaved = sequentialTime - best.time;
    double percentSaved = (sequentialTime > 0) ? (timeSaved / sequentialTime) * 100.0 : 0;
    cout << "  Time Saved   : " << fixed << setprecision(4) << timeSaved
         << " seconds (" << fixed << setprecision(1) << percentSaved << "%)" << endl;

    cout << "\n  Key Observations:" << endl;
    cout << "  - Sequential sort uses standard Quicksort (single thread)" << endl;
    cout << "  - Parallel sort creates threads to sort sub-arrays concurrently" << endl;
    cout << "  - Speedup = Sequential Time / Parallel Time" << endl;
    cout << "  - Efficiency = Speedup / Number of Threads (ideal = 1.0)" << endl;
    cout << "  - Thread creation overhead reduces efficiency at high thread counts" << endl;
    cout << endl;
    printLine();
    cout << "  EXPERIMENT COMPLETE" << endl;
    printLine();
    cout << endl;

    return 0;
}
