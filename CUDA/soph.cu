#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro gestion d'erreur CUDA
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error: %s (code %d), line %d, file %s\n",    \
                cudaGetErrorString(err), err, __LINE__, __FILE__);          \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)


__device__ double square_dev(double x) {
    return x * x;
}

// màj thermique

__global__ void updateGrid(double* current, double* newGrid, int gridSize, double dt, double dx, double diffusionCoeff) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = i * gridSize + j;

    if (i > 0 && i < gridSize - 1 && j > 0 && j < gridSize - 1) {
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(i - 1) * gridSize + j] +
            current[(i + 1) * gridSize + j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]) / square_dev(dx);
    }
    else if (i == 0 && j > 0 && j < gridSize - 1) {
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(gridSize - 1) * gridSize + j] +
            current[(i + 1) * gridSize + j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]) / square_dev(dx);
    }
    else if (i == gridSize - 1 && j > 0 && j < gridSize - 1) {
        newGrid[idx] = current[idx] + diffusionCoeff * dt * (
            current[(i - 1) * gridSize + j] +
            current[j] +
            current[i * gridSize + (j - 1)] +
            current[i * gridSize + (j + 1)] -
            4 * current[idx]) / square_dev(dx);
    }
}


//init des données

void initializeGrid(double* grid, int gridSize) {
    for (int i = 0; i < gridSize * gridSize; i++) {
        grid[i] = 0.0;
    }

    for (int i = 0; i < gridSize; i++) {
        grid[i * gridSize + 0] = pow(cos(i * M_PI / (double)gridSize), 2);
        grid[i * gridSize + (gridSize - 1)] = pow(sin(i * M_PI / (double)gridSize), 2);
    }
}




int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <gridSize>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int gridSize = atoi(argv[1]);
    const double diffusionCoeff = 1.0;
    double dx = M_PI / gridSize;
    double dt = dx * dx / (8.0 * diffusionCoeff);
    double totalTime = 0.5 * M_PI * M_PI / diffusionCoeff;
    int numSteps = (int)(totalTime / dt);
    size_t size = gridSize * gridSize * sizeof(double);

    printf("Taille de la grille : %d x %d\n", gridSize, gridSize);
    printf("Nombre de pas de temps : %d\n", numSteps);

    // allocation CPU
    double* h_current = (double*)malloc(size);
    double* h_new = (double*)malloc(size);
    if (!h_current || !h_new) {
        fprintf(stderr, "Erreur allocation CPU\n");
        return EXIT_FAILURE;
    }

    initializeGrid(h_current, gridSize);
    memcpy(h_new, h_current, size);

    // allocation GPU
    double *d_current, *d_new;
    CUDA_CHECK(cudaMalloc(&d_current, size));
    CUDA_CHECK(cudaMalloc(&d_new, size));
    CUDA_CHECK(cudaMemcpy(d_current, h_current, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice));
    dim3 blockDim(16, 16);
    dim3 gridDim((gridSize + 15) / 16, (gridSize + 15) / 16);
    // mesurer du temps GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int step = 0; step < numSteps; step++) {
        updateGrid<<<gridDim, blockDim>>>(d_current, d_new, gridSize, dt, dx, diffusionCoeff);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        double* temp = d_current;
        d_current = d_new;
        d_new = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Temps GPU total : %.3f ms\n", milliseconds);
    //copie résultats GPU vers CPU
    CUDA_CHECK(cudaMemcpy(h_current, d_current, size, cudaMemcpyDeviceToHost));
    //clcul température moyenne
    double avgTemp = 0.0;
    for (int i = 0; i < gridSize * gridSize; i++) {
        avgTemp += h_current[i];
    }
    avgTemp /= (gridSize * gridSize);
    printf("Température moyenne : %f\n", avgTemp);

    //Écriture dans le fichier
    char filename[64];
    snprintf(filename, sizeof(filename), "map_cuda_%d.txt", gridSize);
    FILE* out = fopen(filename, "w");
    if (out) {
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                fprintf(out, "%d %d %f\n", i, j, h_current[i * gridSize + j]);
            }
            fprintf(out, "\n");
        }
        fclose(out);
    } else {
        fprintf(stderr, "Erreur ouverture fichier sortie\n");
    }

    //libération ressources
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_new));
    free(h_current);
    free(h_new);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
