#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 100        // Matrix size 
#define RUNS 5        // Number of runs to average

// Matrices 
double A[N][N], B[N][N], C[N][N], C_baseline[N][N];

//==================================================================================
//========================= Initialze the matrices with random variables ===========
//==================================================================================
void init_matrices() {
    srand(42); // Fixed seed (Deterministic)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(rand() % 10 + 1); // to give a value between 1 and 10 
            B[i][j] = (double)(rand() % 10 + 1);
        }
}

//==================================================================================
//========================= Baseline (Single-threaded) =============================
//==================================================================================
void baseline_multiply() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C_baseline[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C_baseline[i][j] += A[i][k] * B[k][j];
        }
}

//==================================================================================
//========================= Row-wise Parallelism  ==================================
//==================================================================================
// Each Thread has a struct that have it's start and end row that handles
typedef struct {
    int start_row;  // First row this thread handles
    int end_row;    // Last row 
} Thread_Args;

// void* arg-> pointer to anything (any type), and here is Thread_Args struce
void* row_worker(void* arg) {
    Thread_Args* args = (Thread_Args*)arg;
    int i = args -> start_row;
    int e = args -> end_row;
    for (i; i < e; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return NULL;
}

void row_wise(int threads_num) {
    pthread_t threads[threads_num]; // threads[0] → ID card for thread 0
    Thread_Args args[threads_num]; // one Thread_Args struct per thread

    int rows_per_thread = N / threads_num; // N: total rows in the matrix, threads_num: number of threads 

    for (int t = 0; t < threads_num; t++) {
        args[t].start_row = t * rows_per_thread;// t=0: start_row = 0 * 2 = 0, t=1: start_row = 1 * 2 = 2
        args[t].end_row = (t == threads_num - 1) ? N : (t + 1) * rows_per_thread; // last thread gets any reamaining rows
        pthread_create(&threads[t], NULL, row_worker, &args[t]);
    }

    // Wait for all threads to finish
    for (int t = 0; t < threads_num; t++)
        pthread_join(threads[t], NULL);
}

//==================================================================================
//========================= Column-wise Parallelism ================================
//==================================================================================
typedef struct {
    int start_col;
    int end_col;
} Col_Args;

void* col_worker(void* arg) {
    Col_Args* args = (Col_Args*)arg;
    for (int i = 0; i < N; i++)
        for (int j = args->start_col; j < args->end_col; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return NULL;
}

void colwise_multiply(int threads_num) {
    pthread_t threads[threads_num];
    Col_Args args[threads_num];

    int cols_per_thread = N / threads_num;

    for (int t = 0; t < threads_num; t++) {
        args[t].start_col = t * cols_per_thread;
        args[t].end_col = (t == threads_num - 1) ? N : (t + 1) * cols_per_thread; // last thread gets leftover columns
        pthread_create(&threads[t], NULL, col_worker, &args[t]);
    }

    for (int t = 0; t < threads_num; t++)
        pthread_join(threads[t], NULL);
}

//==================================================================================
//========================= Block (Tiled) Multiplication ==========================
//==================================================================================
typedef struct {
    int row_start, row_end;
    int col_start, col_end;
} Block_Args;

void* block_worker(void* arg) {
    Block_Args* args = (Block_Args*)arg;
    for (int i = args->row_start; i < args->row_end; i++)
        for (int j = args->col_start; j < args->col_end; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    return NULL;
}

void block_multiply(int num_threads) {
    int grid_rows, grid_cols;

    if (num_threads == 2) {
        grid_rows = 1;  
        grid_cols = 2;
    } else if (num_threads == 8) {
        grid_rows = 2; 
        grid_cols = 4;
    } else if (num_threads == 16) {
        grid_rows = 4;  
        grid_cols = 4;
    } else {
        // for any other thread count
        grid_rows = (int)sqrt(num_threads);
        grid_cols = num_threads / grid_rows;
    }

    pthread_t threads[num_threads];
    Block_Args args[num_threads];

    // if N=1000 then block_rows = 1000 / 2 = 500 and block_cols = 1000 / 4 = 250
    int block_rows = N / grid_rows;
    int block_cols = N / grid_cols;

    int t = 0;
    for (int br = 0; br < grid_rows; br++) {
        for (int bc = 0; bc < grid_cols; bc++) {
            args[t].row_start = br * block_rows;
            args[t].row_end   = (br == grid_rows - 1) ? N : (br + 1) * block_rows;
            args[t].col_start = bc * block_cols;
            args[t].col_end   = (bc == grid_cols - 1) ? N : (bc + 1) * block_cols;
            pthread_create(&threads[t], NULL, block_worker, &args[t]);
            t++;
        }
    }
    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);
}

//==================================================================================
//========================= Timing and Averaging ===================================
//==================================================================================
double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double measure_average(void (*func)(int), int threads) {
    double total = 0.0;
    for (int r = 0; r < RUNS; r++) {
        double start = get_time_seconds();
        func(threads);
        double end = get_time_seconds();
        total += (end - start);
    }
    return total / RUNS;
}

//==================================================================================
//========================= Verify Correctness =====================================
//==================================================================================
int verify(double C_parallel[N][N], double C_ref[N][N]) {
    double tolerance = 1e-6;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (fabs(C_parallel[i][j] - C_ref[i][j]) > tolerance)
                return 0; 
    return 1; 
}

//==================================================================================
//========================= Main ===================================================
//==================================================================================
int main() {
    init_matrices();

    // ─── Baseline ───
    double t_start = get_time_seconds();
    baseline_multiply();
    double t_baseline = get_time_seconds() - t_start;
    printf("Matrix size: %d", N);
    printf("\nBaseline: %.4f seconds\n", t_baseline);

    int thread_counts[] = {2, 8, 16};


// ─── Row-wise ───
for (int i = 0; i < 3; i++) {
    int T = thread_counts[i];
    double t = measure_average(row_wise, T);
    printf("Row-wise, %d threads: %.4f sec, speedup: %.2fx\n",
           T, t, t_baseline / t);
    row_wise(T);
    printf("Row-wise %d threads correct: %s\n",
           T, verify(C, C_baseline) ? "YES" : "NO");
//                 
}

// ─── Col-wise ───
for (int i = 0; i < 3; i++) {
    int T = thread_counts[i];
    double t = measure_average(colwise_multiply, T);
    printf("Col-wise, %d threads: %.4f sec, speedup: %.2fx\n",
           T, t, t_baseline / t);
    colwise_multiply(T);
    printf("Col-wise %d threads correct: %s\n",
           T, verify(C, C_baseline) ? "YES" : "NO");
//                 
}

// ─── Block ───
for (int i = 0; i < 3; i++) {
    int T = thread_counts[i];
    double t = measure_average(block_multiply, T);
    printf("Block, %d threads: %.4f sec, speedup: %.2fx\n",
           T, t, t_baseline / t);
    block_multiply(T);
    printf("Block %d threads correct: %s\n",
           T, verify(C, C_baseline) ? "YES" : "NO");
//                  
}
}
