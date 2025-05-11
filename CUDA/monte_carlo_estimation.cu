#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>  // we need it for test tp

#define SAMPLES_PER_THREAD 4096
#define NUM_BLOCKS 256
#define NUM_THREADS 256
#define PI_REFERENCE 3.1415926535f

// ----------------------------------------------------------------------------
// GPU Kernel: Global Memory Version
// each thread computes its Monte Carlo estimate of π and writes the result to a global array.
__global__ void monte_carlo_kernel_global(float *output_estimates, curandState *rng_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the curand state using the seed and the thread id.
    curand_init(1234, tid, 0, &rng_states[tid]);
    //number of rnadom points inside the circle
    int count_inside = 0;
    for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
        float x = curand_uniform(&rng_states[tid]);
        float y = curand_uniform(&rng_states[tid]);
        if (x*x + y*y <= 1.0f) {
            count_inside++;
        }
    }
    output_estimates[tid] = 4.0f * ((float)count_inside / SAMPLES_PER_THREAD);
}

// ----------------------------------------------------------------------------
// GPU Kernel: Shared Memory Version
// Each thread computes its local count, then we reduce the counts in shared memory per block. The first thread in each block computes the block's π estimate.
// plus performante de la globale car elle est rapide
__global__ void monte_carlo_kernel_shared(float *block_estimates, curandState *rng_states) {
    extern __shared__ int shared_counts[]; // dynamic shared memory: one int per thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread creates its own local curand state.
    curandState localState;
    curand_init(1234, tid, 0, &localState);

    int local_count = 0;
    for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
        float x = curand_uniform(&localState);
        float y = curand_uniform(&localState);
        if (x*x + y*y <= 1.0f)
            local_count++;
    }

    // Store each thread's count into shared memory.
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    // Parallel reduction in shared memory to sum the counts for this block.
    // Note: reduction works by halving the number of active threads each iteration.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The block's first thread writes the block estimate computed from its total count.
    if (threadIdx.x == 0) {
        int block_inside = shared_counts[0];
        int total_block_samples = blockDim.x * SAMPLES_PER_THREAD;
        block_estimates[blockIdx.x] = 4.0f * ((float)block_inside / total_block_samples);
    }
}

// ----------------------------------------------------------------------------
// CPU-Side Monte Carlo Estimation of π
float estimate_pi_cpu(long total_samples) {
    float x_coord, y_coord;
    long inside_circle = 0;
    for (long i = 0; i < total_samples; i++) {
        x_coord = rand() / (float)RAND_MAX;
        y_coord = rand() / (float)RAND_MAX;
        if (x_coord*x_coord + y_coord*y_coord <= 1.0f)
            inside_circle++;
    }
    return 4.0f * inside_circle / total_samples;
}

// ----------------------------------------------------------------------------
// Main function
int main(int argc, char *argv[]) {
    // Create CUDA events for timing GPU executions.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_threads = NUM_BLOCKS * NUM_THREADS;

    // ------------------------------------------
    // GPU Global Memory Version
    // ------------------------------------------
    // Allocate GPU memory for the output and random states.
    float *d_results_global;
    cudaMalloc((void **)&d_results_global, total_threads * sizeof(float));

    curandState *d_rng_states;
    cudaMalloc((void **)&d_rng_states, total_threads * sizeof(curandState));

    // Record the start event on stream0 and launch the global kernel.
    cudaEventRecord(start);
    monte_carlo_kernel_global<<<NUM_BLOCKS, NUM_THREADS, 0>>>(d_results_global, d_rng_states);
    cudaEventRecord(stop);

    float elapsed_global = 0.0f;
    cudaEventElapsedTime(&elapsed_global, start, stop);

    // Copy the results from GPU to host.
    float *h_results_global = (float*)malloc(total_threads * sizeof(float));
    cudaMemcpy(h_results_global, d_results_global, total_threads * sizeof(float), cudaMemcpyDeviceToHost);

    // Average the per-thread estimates to get an overall π estimate.
    float pi_gpu_global = 0.0f;
    for (int i = 0; i < total_threads; i++) {
        pi_gpu_global += h_results_global[i];
    }
    pi_gpu_global /= total_threads;

    printf("GPU Global Memory version:\n");
    printf("  Estimated PI = %f\n", pi_gpu_global);
    printf("  Error = %f\n", pi_gpu_global - PI_REFERENCE);
    printf("  Kernel execution time = %f ms\n", elapsed_global);

    // ------------------------------------------
    // GPU Shared Memory Version
    // ------------------------------------------
    // Allocate GPU memory for the block-level results.
    float *d_block_results;
    cudaMalloc((void **)&d_block_results, NUM_BLOCKS * sizeof(float));

    // Launch the shared-memory kernel in stream1.
    // Here, the third kernel parameter sets the dynamic shared memory size.
    cudaEventRecord(start);
    monte_carlo_kernel_shared<<<NUM_BLOCKS, NUM_THREADS, NUM_THREADS * sizeof(int)>>>(d_block_results, d_rng_states);
    cudaEventRecord(stop);

    float elapsed_shared = 0.0f;
    cudaEventElapsedTime(&elapsed_shared, start, stop);

    // Copy block results from GPU to host.
    float *h_block_results = (float*)malloc(NUM_BLOCKS * sizeof(float));
    cudaMemcpy(h_block_results, d_block_results, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    // Average the block estimates.
    float pi_gpu_shared = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        pi_gpu_shared += h_block_results[i];
    }
    pi_gpu_shared /= NUM_BLOCKS;

    printf("GPU Shared Memory version:\n");
    printf("  Estimated PI = %f\n", pi_gpu_shared);
    printf("  Error = %f\n", pi_gpu_shared - PI_REFERENCE);
    printf("  Kernel execution time = %f ms\n", elapsed_shared);

    // ------------------------------------------
    // Cleanup GPU memory and events/streams.
    cudaFree(d_results_global);
    cudaFree(d_rng_states);
    cudaFree(d_block_results);
    free(h_results_global);
    free(h_block_results);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ------------------------------------------
    // CPU Version Execution and Timing
    // ------------------------------------------
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    float pi_cpu = estimate_pi_cpu(total_threads * SAMPLES_PER_THREAD);
    cpu_end = clock();

    float elapsed_cpu = (cpu_end - cpu_start) / (float)CLOCKS_PER_SEC;

    printf("CPU version:\n");
    printf("  Estimated PI = %f\n", pi_cpu);
    printf("  Error = %f\n", pi_cpu - PI_REFERENCE);
    printf("  Execution time = %f seconds\n", elapsed_cpu);

    return 0;
}
