#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/* Structure to store vector data and result */
typedef struct {
    double *vectorA;
    double *vectorB;
    double result;
    int vectorLength;
} DOTDATA;

#define VECTOR_SIZE 100000
DOTDATA dotProductData;

/* Function to compute dot product */
void compute_dot_product() {
    int index_start, index_end, i;
    double local_sum = 0, *inputA, *inputB;

    index_start = 0;
    index_end = dotProductData.vectorLength;
    inputA = dotProductData.vectorA;
    inputB = dotProductData.vectorB;

    #pragma omp parallel for reduction(+:local_sum) private(i)
    for (i = index_start; i < index_end; i++) {
        local_sum += inputA[i] * inputB[i];
    }

    /**
    #pragma omp parallel for shared(local_sum)  
    for (i = index_start; i < index_end; i++) {  
       local_sum += inputA[i] * inputB[i];  
    }  
    for simd schedule(static)
    */
    dotProductData.result = local_sum;
}

int main(int argc, char *argv[]) {
    int i, length;
    double *vectorX, *vectorY;
    clock_t start, end;

    length = VECTOR_SIZE;
    vectorX = (double *)malloc(length * sizeof(double));
    vectorY = (double *)malloc(length * sizeof(double));

    for (i = 0; i < length; i++) {
        vectorX[i] = 1.0;
        vectorY[i] = vectorX[i];
    }

    dotProductData.vectorLength = length;
    dotProductData.vectorA = vectorX;
    dotProductData.vectorB = vectorY;
    dotProductData.result = 0;

    start = omp_get_wtime();  // Start timing
    compute_dot_product();
    end = omp_get_wtime();  // End timing

    printf("OpenMP (reduction) - Dot Product Sum = %f\n", dotProductData.result);
    printf("OpenMP Version - Execution Time = %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    free(vectorX);
    free(vectorY);
    return 0;
}
