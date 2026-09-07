# Multithreaded Matrix Multiplication with Performance Analysis


This project multiplies two square `N x N` matrices of `double` values using four different strategies — a single-threaded baseline plus three POSIX-threads parallelizations — and measures how each one performs against the baseline.

## Files

| File | Description |
| --- | --- |
| `Multithreaded-Matrix-Multiplication(OS).c` | Full implementation: all four approaches, timing harness, and correctness verification. |
| `README.md` | This file. |

The written report (programming environment, test cases, performance analysis, and conclusion) is submitted separately as `Project1.pdf`.

## Building and running

The program uses POSIX threads and `clock_gettime`, so it needs a Unix-like environment. It was developed and tested on Ubuntu under WSL2.

```bash
gcc -O2 -o matmul "Multithreaded-Matrix-Multiplication(OS).c" -lpthread -lm
./matmul
```

`-lm` is required because the block partitioning falls back to `sqrt()` for thread counts other than 2, 8, and 16.

The matrix size is a compile-time constant, so changing it means editing the `#define` at the top of the source file and recompiling:

```c
#define N 100   // change to 500 or 1000 to reproduce the other test cases
#define RUNS 5  // number of timed runs that get averaged
```

Running the program prints the baseline time, then for each approach and each thread count (2, 8, 16) the average time, the speedup over the baseline, and whether the result matched the baseline.

## How it works

**Matrix setup.** `A`, `B`, `C`, and `C_baseline` are global `double[N][N]` arrays. Declaring them globally means every thread can reach them directly through the process's shared address space instead of receiving them as arguments. `A` and `B` are filled with `rand() % 10 + 1` (values from 1 to 10) after `srand(42)`, so every run uses identical input data.

**Thread arguments.** A pthread worker function can only accept a single `void*`, so each approach bundles its parameters into a small struct (`Thread_Args`, `Col_Args`, `Block_Args`) and passes a pointer to it. The worker casts the pointer back to the right struct type.

**No locking.** Every thread writes to a disjoint region of `C`, so no two threads ever touch the same memory location and no mutexes are needed during computation. This matters for performance, since locking would add both overhead and serialization.

### The four approaches

**Baseline (single-threaded)** — the standard triple-nested loop. Its result is stored once in `C_baseline` and never modified again, which makes it the reference for both correctness and speedup.

**Row-wise parallelism** — `C` is split horizontally into groups of consecutive rows. Each thread gets a `start_row`/`end_row` pair and computes every column for its rows. Each thread receives `N / threads_num` rows, and the last thread absorbs any remainder. Because C stores matrices in row-major order, walking `C[i][j]` with increasing `j` is sequential in memory, which uses cache lines well.

**Column-wise parallelism** — the same idea applied vertically: each thread owns a `start_col`/`end_col` range and loops over all rows. Cache behaviour is worse here, because for a fixed column the step from `C[i][j]` to `C[i+1][j]` jumps `N` elements forward in memory.

**Block (tiled) multiplication** — `C` is divided into a 2D grid of rectangular tiles, one per thread, each defined by both a row range and a column range. The grid shape is chosen from the thread count:

| Threads | Grid |
| --- | --- |
| 2 | 1 × 2 |
| 8 | 2 × 4 |
| 16 | 4 × 4 |

The last block in each direction extends to `N` so that sizes not evenly divisible by the grid dimensions are still fully covered.

### Timing and verification

`get_time_seconds()` wraps `clock_gettime(CLOCK_MONOTONIC)` and returns seconds as a `double`, so elapsed time is a plain subtraction. `measure_average()` runs each configuration `RUNS` (5) times and reports the mean, which smooths out transient system load.

After timing, each approach is run once more and `verify()` compares every cell of `C` against `C_baseline` with a tolerance of `1e-6` to absorb floating-point differences. All approaches passed verification at every matrix size and thread count.

## Results

Measured on 16 logical CPU cores (Ubuntu on WSL2, GCC with `-O2`), averaged over 5 runs.

| Matrix size | Baseline time | Best speedup (16 threads, block) |
| --- | --- | --- |
| 100 × 100 | 0.0004 s | none — all parallel approaches were slower than or equal to the baseline |
| 500 × 500 | 0.0562 s | 6.41× |
| 1000 × 1000 | 0.6049 s | 3.80× |

Speedups at 16 threads, by approach:

| Matrix size | Row-wise | Column-wise | Block |
| --- | --- | --- | --- |
| 500 × 500 | 5.76× | 6.26× | 6.41× |
| 1000 × 1000 | 3.42× | 3.69× | 3.80× |

At 100 × 100 the computation finishes so quickly that thread creation and join overhead outweighs any benefit from running in parallel. From 500 × 500 onward every parallel approach beats the baseline. The three parallel strategies land close to each other, which is expected — they perform exactly the same arithmetic and differ only in memory access pattern. Adding threads helps with diminishing returns: 2 → 8 threads is a large jump, 8 → 16 much smaller.

None of the approaches reach the theoretical 16× because matrix multiplication is memory-bound at this scale. The main limiters are memory bandwidth contention between threads, cache pressure from many threads competing for shared cache, thread creation overhead, the non-parallelizable portions of the program (initialization, timing, verification) per Amdahl's Law, and slight load imbalance when `N` is not divisible by the thread count. The full discussion is in the report.

## Notes and limitations

- `N` and `RUNS` are compile-time constants; each matrix size requires a separate build.
- The block grid is hard-coded for 2, 8, and 16 threads. Other counts fall back to `sqrt(num_threads)` for the grid rows, which for non-factorable counts creates fewer threads than requested while still joining `num_threads` handles.
- Timing covers only the multiplication itself, not matrix initialization or verification.
