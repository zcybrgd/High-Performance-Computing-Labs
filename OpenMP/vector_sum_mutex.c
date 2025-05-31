#include <stdio.h>
#include <pthread.h>

#define SIZE 256
#define NUM_THREADS 4

double a[SIZE], b[SIZE];
double sum = 0.0;
pthread_mutex_t sum_mutex;

// Structure to pass arguments to threads
typedef struct {
    int thread_id;
} thread_data_t;

void* compute_partial_sum(void* arg) {
    thread_data_t* data = (thread_data_t*) arg;
    int tid = data->thread_id;

    // Determine the range of indices this thread will work on
    int start = tid * (SIZE / NUM_THREADS);
    int end = (tid + 1) * (SIZE / NUM_THREADS);

    double local_sum = 0.0;
    for (int i = start; i < end; i++)
        local_sum += a[i] * b[i];

    // Update global sum with mutex protection
    pthread_mutex_lock(&sum_mutex);
    sum += local_sum;
    pthread_mutex_unlock(&sum_mutex);

    pthread_exit(NULL);
}

int main() {
    // Initialize vectors
    for (size_t i = 0; i < SIZE; i++) {
        a[i] = i * 0.5;
        b[i] = i * 2.0;
    }

    // Initialize mutex
    pthread_mutex_init(&sum_mutex, NULL);

    // Create threads
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        pthread_create(&threads[i], NULL, compute_partial_sum, (void*) &thread_data[i]);
    }

    // Join threads
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    // Destroy mutex
    pthread_mutex_destroy(&sum_mutex);

    // Print final result
    printf("sum = %g\n", sum);

    return 0;
}
