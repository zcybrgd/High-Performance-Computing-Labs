#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

long long sumOfSquaresParallelCritical(int* array, int size);

int main() {
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    clock_t start = clock();
    long long total = sumOfSquaresParallelCritical(array, ARRAY_SIZE);
    clock_t end = clock();

    printf("Parallel (Reduction) Total: %lld\n", total);
    printf("Time taken (Reduction): %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(array);
    return 0;
}

long long sumOfSquaresParallelCritical(int* array, int size) {
    long long total = 0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        long long square = array[i] * array[i];
        //pour que we make sure que un seul thread accede a la variable totale pour ajouter sa somme , un thread ela darba so twli kichghoul parallel
        #pragma omp critical
        total += square;
    }
    return total;
}
