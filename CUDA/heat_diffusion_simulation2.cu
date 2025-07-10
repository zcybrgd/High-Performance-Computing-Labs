%%writefile par2.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

__device__ double square_dev(double x) {
    return x * x;
}

__global__ void updateGrid(double* current, double* newGrid, int gridSize, double dt, double dx, double diffusionCoeff) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;



    if (i > 0 && i < gridSize - 1 && j > 0 && j < gridSize - 1) {
        int idx = i * gridSize + j;
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(i - 1) * gridSize + j] +
            current[(i + 1) * gridSize + j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]) / square_dev(dx);
    }
    else if (i == 0 && j > 0 && j < gridSize - 1) {
        int idx = i * gridSize + j;
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(gridSize - 1) * gridSize + j] +
            current[(i + 1) * gridSize + j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]
        ) / square_dev(dx);
    }
    else if (i == gridSize - 1 && j > 0 && j < gridSize - 1) {
        int idx = i * gridSize + j;
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(i - 1) * gridSize + j] +
            current[0 * gridSize + j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]
        ) / square_dev(dx);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    const int gridSize = atoi(argv[1]);
    const double diffusionCoeff = 1;
    double dx = M_PI / gridSize;
    double dt = dx * dx / (8 * diffusionCoeff);
    double totalTime = 0.5 * M_PI * M_PI / diffusionCoeff;
    int numSteps = (int)(totalTime / dt);
    size_t size = gridSize * gridSize * sizeof(double);

    double* currentGrid = (double*)malloc(size);
    double* newGrid = (double*)malloc(size);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            currentGrid[i * gridSize + j] = 0;
        }
    }

    for (int i = 0; i < gridSize; i++) {
        currentGrid[i * gridSize + 0] = pow(cos(i * M_PI / (double)gridSize), 2);
        currentGrid[i * gridSize + (gridSize - 1)] = pow(sin(i * M_PI / (double)gridSize), 2);
    }

    memcpy(newGrid, currentGrid, size);

    // CUDA memory allocation
    double *d_currentGrid, *d_newGrid;
    checkCudaError(cudaMalloc(&d_currentGrid, size), "cudaMalloc currentGrid");
    checkCudaError(cudaMalloc(&d_newGrid, size), "cudaMalloc newGrid");

    checkCudaError(cudaMemcpy(d_currentGrid, currentGrid, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D current");
    checkCudaError(cudaMemcpy(d_newGrid, newGrid, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D new");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + 15) / 16, (gridSize + 15) / 16);

    clock_t start = clock();
    for (int step = 0; step < numSteps; step++) {
        updateGrid<<<numBlocks, threadsPerBlock>>>(d_currentGrid, d_newGrid, gridSize, dt, dx, diffusionCoeff);
        cudaDeviceSynchronize();
        double* temp = d_currentGrid;
        d_currentGrid = d_newGrid;
        d_newGrid = temp;
    }
    clock_t end = clock();

    checkCudaError(cudaMemcpy(currentGrid, d_currentGrid, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    double averageTemp = 0.0;
    for (int i = 0; i < gridSize * gridSize; i++) {
        averageTemp += currentGrid[i];
    }

    printf("Total time (parallel) = %f seconds\n", ((float)(end - start)) / CLOCKS_PER_SEC);
    printf("Average temperature = %f\n", averageTemp / (gridSize * gridSize));

    // Write output
    FILE* fileOut;
    char filename[50];
    sprintf(filename, "map_cuda_%d.txt", gridSize);
    fileOut = fopen(filename, "w");
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            fprintf(fileOut, "%d %d %f\n", i, j, currentGrid[i * gridSize + j]);
        }
        fprintf(fileOut, "\n");
    }
    fclose(fileOut);

    free(currentGrid);
    free(newGrid);
    cudaFree(d_currentGrid);
    cudaFree(d_newGrid);
    return 0;
}
