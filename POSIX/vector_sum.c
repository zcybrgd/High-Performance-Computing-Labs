#include <stdio.h>
#include <pthread.h>

#define SIZE 256
#define NUM_THREADS 4

double a[SIZE], b[SIZE];
double partial_sums[NUM_THREADS];

// Structure pour passer les bornes et l'index du thread
typedef struct {
    int thread_id;
    int start;
    int end;
} thread_data_t;

void* compute_partial_sum(void* arg) {
    thread_data_t* data = (thread_data_t*) arg;
    int start = data->start;
    int end = data->end;
    int tid = data->thread_id;

    double local_sum = 0.0;
    for (int i = start; i < end; i++)
        local_sum += a[i] * b[i];

    partial_sums[tid] = local_sum;
    pthread_exit(NULL);

    /**
    double* result = malloc(sizeof(double));
    *result = local_sum;
    pthread_exit(result);
    */
}

int main() {
    // Initialiser les vecteurs
    for (size_t i = 0; i < SIZE; i++) {
        a[i] = i * 0.5;
        b[i] = i * 2.0;
    }

    // Initialiser les sous-totaux
    for (int i = 0; i < NUM_THREADS; i++)
        partial_sums[i] = 0.0;

    // Créer les threads
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];

    int chunk_size = SIZE / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == NUM_THREADS - 1) ? SIZE : (i + 1) * chunk_size;

        pthread_create(&threads[i], NULL, compute_partial_sum, (void*) &thread_data[i]);
    }

    // Attendre la fin des threads
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
  /**
        pthread_join(threads[i], &ret);
        sum += *(double*)ret;
        free(ret);
  */

    // Somme finale
    double sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++)
        sum += partial_sums[i];

    printf("sum = %g\n", sum);

    return 0;
}
