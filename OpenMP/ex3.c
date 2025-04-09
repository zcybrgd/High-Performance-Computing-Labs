#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000000

// Function prototype
long long sumOfSquaresSequential(int* array, int size);

int main() {
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    // Populate the array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 100;
    }

    clock_t start = clock();
    long long total = sumOfSquaresSequential(array, ARRAY_SIZE);
    clock_t end = clock();

    printf("Sequential Total: %lld\n", total);
    printf("Time taken (Sequential): %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(array);
    return 0;
}

long long sumOfSquaresSequential(int* array, int size) {
    long long total = 0;
    for (int i = 0; i < size; i++) {
        total += array[i] * array[i];
    }
    return total;
}

//lets turn it parallel : once with reduction, once  with criticial, once with  atomic and explain the code 
