#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

int main(int argc, char* argv[]) {
    int rank, size;
    double A[N][N], X[N], y_local, Y[N];
    double row[N];  // Pour recevoir sa ligne

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0)
            printf("Erreur: ce programme nécessite exactement %d processus.\n", N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        // Initialiser A et X
        printf("Matrice A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j + 1;
                printf("%5.1f ", A[i][j]);
            }
            printf("\n");
        }

        printf("Vecteur X:\n");
        for (int i = 0; i < N; i++) {
            X[i] = i + 1;
            printf("%5.1f\n", X[i]);
        }
    }

    // Diffuser X à tous les processus
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribuer une ligne de A à chaque processus
    MPI_Scatter(A, N, MPI_DOUBLE, row, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcul local
    y_local = 0.0;
    for (int j = 0; j < N; j++)
        y_local += row[j] * X[j];

    // Rassembler les résultats
    MPI_Gather(&y_local, 1, MPI_DOUBLE, Y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Afficher le résultat final
    if (rank == 0) {
        printf("Résultat Y = A * X :\n");
        for (int i = 0; i < N; i++)
            printf("%5.1f\n", Y[i]);
    }

    MPI_Finalize();
    return 0;
}
