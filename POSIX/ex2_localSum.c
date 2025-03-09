
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define SIZE 100000000 // Taille du tableau
#define NUM_THREADS 4  // Nombre de threads

typedef struct {
    int *array;
    long long sum;
    int start, end;
} ThreadData;

void *sumArrayParallel(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    data->sum = 0;
    for (int i = data->start; i < data->end; i++) {
        data->sum += data->array[i];
    }
    pthread_exit(NULL);
}

int main() {
    int *array = (int *)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; i++) {
        array[i] = i;
    }

   
    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    int chunk_size = SIZE / NUM_THREADS;
    clock_t start = clock();
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].array = array;
        threadData[i].start = i * chunk_size;
        threadData[i].end = (i == NUM_THREADS - 1) ? SIZE : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, sumArrayParallel, &threadData[i]);
    }

    long long sum_parallel = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        sum_parallel += threadData[i].sum;
    }
    clock_t end = clock();
    double time_parallel = (double)(end - start) / CLOCKS_PER_SEC;

    // Affichage des résultats
    printf("Somme parallèle: %lld\n", sum_parallel);
    printf("Temps parallèle: %f secondes\n", time_parallel);

    free(array);
    return 0;
}