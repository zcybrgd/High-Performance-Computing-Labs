
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

// Function prototypes
long long sumOfSquaresSequential(int* array, int size);
long long sumOfSquaresReduction(int* array, int size);
long long sumOfSquaresCritical(int* array, int size);
long long sumOfSquaresAtomic(int* array, int size);

int main() {
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    double start, end;

    // Sequential
    start = omp_get_wtime();
    long long total_seq = sumOfSquaresSequential(array, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Sequential Total: %lld\n", total_seq);
    printf("Time taken (Sequential): %f seconds\n\n", end - start);

    // OpenMP with reduction
    start = omp_get_wtime();
    long long total_red = sumOfSquaresReduction(array, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Reduction Total: %lld\n", total_red);
    printf("Time taken (Reduction): %f seconds\n\n", end - start);

    // OpenMP with critical
    start = omp_get_wtime();
    long long total_crit = sumOfSquaresCritical(array, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Critical Total: %lld\n", total_crit);
    printf("Time taken (Critical): %f seconds\n\n", end - start);

    // OpenMP with atomic
    start = omp_get_wtime();
    long long total_atom = sumOfSquaresAtomic(array, ARRAY_SIZE);
    end = omp_get_wtime();
    printf("Atomic Total: %lld\n", total_atom);
    printf("Time taken (Atomic): %f seconds\n\n", end - start);

    free(array);
    return 0;
}

// Sequential version
long long sumOfSquaresSequential(int* array, int size) {
    long long total = 0;
    for (int i = 0; i < size; i++) {
        total += array[i] * array[i];
    }
    return total;
}

// Parallel version using reduction
long long sumOfSquaresReduction(int* array, int size) {
    long long total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < size; i++) {
        total += array[i] * array[i];
    }
    return total;
}

// Parallel version using critical section
long long sumOfSquaresCritical(int* array, int size) {
    long long total = 0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        long long square = array[i] * array[i];
        #pragma omp critical
        {
            total += square;
        }
    }
    return total;
}

// Parallel version using atomic
long long sumOfSquaresAtomic(int* array, int size) {
    long long total = 0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        long long square = array[i] * array[i];
        #pragma omp atomic
        total += square;
    }
    return total;
}