#define N 512

__global__ void dot(int *a, int *b, int *c) {
    __shared__ int temp[N]; // mémoire partagée

    // chaque thread calcule un produit partiel
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

    __syncthreads(); // on attend que tous les threads aient fini

    // seul le thread 0 fait la somme
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < N; i++)
            sum += temp[i];
        *c = sum;
    }
}



int main(void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(int);

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, sizeof(int));

    a = (int*)malloc(size);
    b = (int*)malloc(size);

    random_ints(a, N);
    random_ints(b, N);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dot<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Nettoyage
    free(a); free(b); free(c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 0;
}
