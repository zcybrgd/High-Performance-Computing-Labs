#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

long long sumOfSquaresParallelReduction(int* array, int size);

int main() {
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    clock_t start = clock();
    long long total = sumOfSquaresParallelReduction(array, ARRAY_SIZE);
    clock_t end = clock();

    printf("Parallel (Reduction) Total: %lld\n", total);
    printf("Time taken (Reduction): %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(array);
    return 0;
}

long long sumOfSquaresParallelReduction(int* array, int size) {
    long long total = 0;
    // chaque thread calcule une somme locale et a la toute fin cette pragma prend en charge la somme totale
    // pas de mutex et pas de besoin de synchro
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < size; i++) {
        total += array[i] * array[i];
    }
    return total;
}
