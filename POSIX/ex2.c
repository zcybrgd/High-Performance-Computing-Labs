#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#define SIZE 100000000 // array size
#define num_threads 8

typedef struct {
    long long *array;
    long long start;
    long long end;
} ThreadArgs;


// Function for calculating the sum of the elements in an array
long long sumArray(long long *array, long long size) {
    long long sum = 0;
    for (long long i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}
// the parallel version to be entered
void* sumArrayParallel(void * args) {
    ThreadArgs* arguments = (ThreadArgs*)args;
    long long sum = 0;
    for (long long i = arguments->start; i < arguments->end; i++) {
        sum += arguments->array[i];
    }
    pthread_exit((void*)(intptr_t)sum);
}

int main() {
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];
    long long chunk_size = SIZE / num_threads;
    long long *array = (long long *)malloc(SIZE * sizeof(long long));
    // Fill in the table with values
    for (long long i = 0; i < SIZE; i++) {
        array[i] = i;
    }

    for(int i=0; i<num_threads; i++){
       args[i].array = array;
       args[i].start = i* chunk_size;
       args[i].end = (i == num_threads - 1) ? SIZE : (i + 1) * chunk_size;

       if (pthread_create(&threads[i], NULL, sumArrayParallel, (void*)&args[i]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            free(array);
            exit(EXIT_FAILURE);
        }
    }

    clock_t start = clock();
    long long sum = sumArray(array, SIZE);
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sum: %lld\n", sum);
    printf("Sequential time: %f seconds\n", time_taken);


     // Measure parallel execution time
    clock_t start_par = clock();

    // Collect results from threads
    long long parallel_sum = 0;
    for (int i = 0; i < num_threads; i++) {
        void *valeur;
        if (pthread_join(threads[i], &valeur) != 0) {
            fprintf(stderr, "Error joining thread %d\n", i);
            free(array);
            exit(EXIT_FAILURE);
        }
        parallel_sum += (long long)(intptr_t)valeur;
    }

    clock_t end_par = clock();
    double par_time = (double)(end_par - start_par) / CLOCKS_PER_SEC;
    printf("Parallel Sum: %lld\n", parallel_sum);
    printf("Parallel Time: %f seconds\n", par_time);
    printf("Speedup: %.2f\n", time_taken / par_time);

    free(array);
    return 0;
}


/*
fonctions de temps , integrer le code parallele avec le code sequentiel et remarquer les performances des 2 en mm temps
segmenttion error si nombre de thread depasse
structure de donnes crees pour les faire passer au threads
utiliser les variables locales et chaque thread work independamment, ou variable globa;e, retourner la somme locale avec pthread exit pour la recuperer ensuite avec pthread join
et faire la some globale
*/
