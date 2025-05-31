#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4  // Taille de la matrice et des vecteurs

int main(int argc, char* argv[]) {
    int rank, size;
    double A[N][N];
    double X[N];
    double y_local, Y[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Vérifier que nous avons exactement 4 processus
    if (size != N) {
        if (rank == 0)
            printf("Erreur: ce programme nécessite exactement %d processus.\n", N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialisation par le processus 0
    if (rank == 0) {
        printf("Matrice A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j + 1;  // Par exemple : valeurs simples
                printf("%5.1f ", A[i][j]);
            }
            printf("\n");
        }

        printf("Vecteur X:\n");
        for (int i = 0; i < N; i++) {
            X[i] = i + 1;  // Par exemple : 1, 2, 3, 4
            printf("%5.1f\n", X[i]);
        }
    }

    // Diffuser le vecteur X à tous les processus
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Envoyer la ligne de la matrice A[i] au processus i
    double row[N];
    if (rank == 0) {
        // Envoyer les lignes aux processus 1, 2, 3
        for (int i = 1; i < N; i++)
            MPI_Send(A[i], N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        // Le processus 0 copie sa propre ligne (la ligne 0)
        for (int j = 0; j < N; j++)
            row[j] = A[0][j];
    } else {
        // Recevoir la ligne correspondante
        MPI_Recv(row, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Chaque processus calcule son élément de y
    y_local = 0.0;
    for (int j = 0; j < N; j++)
        y_local += row[j] * X[j];

    // Rassembler les résultats dans Y au processus 0
    MPI_Gather(&y_local, 1, MPI_DOUBLE, Y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Résultat Y = A * X :\n");
        for (int i = 0; i < N; i++)
            printf("%5.1f\n", Y[i]);
    }

    MPI_Finalize();
    return 0;
}
