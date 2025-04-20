#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//nbr de slots=nbr de coeurs
int main(int argc, char** argv)
{
    int rank, size, data;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter an integer to send around the ring: ");
        fflush(stdout);
        scanf("%d", &data);

        // Send to next process
        MPI_Send(&data, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

        // Receive from the last process
        MPI_Recv(&data, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &status);

    } else {
        // Receive from previous process
        MPI_Recv(&data, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

        // Send to next process
        MPI_Send(&data, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    }

    // Print output after receiving
    printf("Hello from rank %d of %d, rank %d has received a message with data %d from rank %d.\n",
           rank, size, rank, data, (rank == 0) ? size - 1 : rank - 1);

    MPI_Finalize();
    return 0;
}
