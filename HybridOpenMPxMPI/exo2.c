#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#define VECTOR_SIZE 100000

//calcule le dot product dans un vecteur de longueur definie
double compute_dot_product(double *vectorA, double *vectorB, int length) {
    double local_sum = 0;
    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < length; i++) {
        local_sum += vectorA[i] * vectorB[i];
    }
    return local_sum;
}

int main(int argc, char *argv[]) {
    int rank, numprocs, chunk_size;
    double *vectorA, *vectorB, local_result = 0, global_result = 0;
    clock_t start, end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    //diviser le vecteur sur le nombre de machines dans le cluster
    chunk_size = VECTOR_SIZE / numprocs;
    vectorA = (double *)malloc(chunk_size * sizeof(double));
    vectorB = (double *)malloc(chunk_size * sizeof(double));

    //init
    for (int i = 0; i < chunk_size; i++) {
        vectorA[i] = 1.0;
        vectorB[i] = 1.0;
    }
    start = clock();
    //chaque machine calcule sans dot product
    local_result = compute_dot_product(vectorA, vectorB, chunk_size);

    //faire la reduction
    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    end = clock();
    // Print result from root process
    if (rank == 0) {
        printf("Parallel Hybrid Version - Dot Product Sum = %f with time of : %f\n", global_result, ((double)(end-start))/ CLOCKS_PER_SEC);
    }

    free(vectorA);
    free(vectorB);
    MPI_Finalize();
    return 0;
}
