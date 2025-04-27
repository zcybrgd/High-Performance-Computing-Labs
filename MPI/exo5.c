#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int world_rank, world_size;
    int color, key;
    MPI_Comm new_comm;
    int new_rank, new_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 4) {
        if (world_rank == 0)
            printf("This program needs exactly 4 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Step 1: Decider la couleur ( 0 pour pair, 1 pour impair)
    if (world_rank % 2 == 0)
        color = 0; // pair -> communicator A
    else
        color = 1; // impair -> communicator B

    // Step 2: la cle (order dans le groupe)
    if (color == 0)
        key = world_rank; // Same order
    else
        key = -world_rank; // Reverse order

    // Step 3: Split the communicator
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

    // Step 4: Get new rank and size in the sub-communicator
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    // Step 5: Display information
    printf("Global rank %d -> New rank %d in communicator %c\n",
           world_rank, new_rank, (color == 0) ? 'A' : 'B');

    MPI_Comm_free(&new_comm);
    MPI_Finalize();
    return 0;
}
//pour assurer la communication entre plusieurs communicateurs
// le comm principale endo les mm processus (il a pas change apres split) donc ils ont des processus en commun
