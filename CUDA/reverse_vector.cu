#include <stdio.h>
#include <cuda_runtime.h>

#define N (2048*2048)  // Taille totale du vecteur
#define BLOCK_SIZE 256 // Taille de bloc (adaptée au GPU)

__global__ void reverse_vector(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[n - 1 - idx] = input[idx];
    }
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocation mémoire sur l'hôte
    h_input  = (float*) malloc(N * sizeof(float));
    h_output = (float*) malloc(N * sizeof(float));

    // Initialiser le vecteur d'entrée
    for (int i = 0; i < N; i++)
        h_input[i] = (float) i;

    // Allocation mémoire sur le GPU
    cudaMalloc((void**)&d_input,  N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copier les données sur le GPU
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculer la configuration d'exécution
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Lancer le kernel
    reverse_vector<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Vérifier quelques valeurs (par exemple les 5 premiers et 5 derniers éléments)
    printf("Premiers éléments du vecteur inversé:\n");
    for (int i = 0; i < 5; i++)
        printf("h_output[%d] = %f\n", i, h_output[i]);

    printf("Derniers éléments du vecteur inversé:\n");
    for (int i = N - 5; i < N; i++)
        printf("h_output[%d] = %f\n", i, h_output[i]);

    // Libérer la mémoire
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
