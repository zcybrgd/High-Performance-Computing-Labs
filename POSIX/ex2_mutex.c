
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>  // Include this for intptr_t

#define SIZE 100000000 // Taille du tableau
#define NUM_THREADS 4  // Nombre de threads

typedef struct {
    int *array;
    int start, end;
} ThreadArgs;

long long sum = 0; // Variable partagée
pthread_mutex_t mutex; // Mutex pour protéger sum

void *sumArrayParallel(void* args) {
    ThreadArgs* argument = (ThreadArgs*)args;
    long long local_sum = 0;

    for (int i = argument->start; i < argument->end; i++) {
        local_sum += argument->array[i];
    }

    pthread_mutex_lock(&mutex);
    sum += local_sum;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main() {
    int *array = (int *)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        array[i] = i;
    }

    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];
    int chunk_size = SIZE / NUM_THREADS;

    pthread_mutex_init(&mutex, NULL); // Initialisation du mutex

    clock_t start = clock();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].array = array;
        args[i].start = i * chunk_size;
        args[i].end = (i == NUM_THREADS - 1) ? SIZE : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, sumArrayParallel, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end = clock();
    double time_parallel = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Somme parallèle: %lld\n", sum);
    printf("Temps parallèle: %f secondes\n", time_parallel);

    pthread_mutex_destroy(&mutex); // Destruction du mutex
    free(array);
    return 0;
}