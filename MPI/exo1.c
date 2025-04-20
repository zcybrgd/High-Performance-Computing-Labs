#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h> // for gethostname
#include <time.h>   // for timing

int main(int argc, char** argv)
{
int myrank, size;
char hostname[MPI_MAX_PROCESSOR_NAME];
int name_len;
double start_time, end_time;

MPI_Init(NULL,NULL);
//get the number of processes launched
MPI_Comm_size(MPI_COMM_WORLD,&size);
//get the rank of the current process
MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
// get the hostname and the length of the name : quelle machine physique execute cette tache
MPI_Get_processor_name(hostname, &name_len);

if (size != 2) {
        if (myrank == 0) {
            fprintf(stderr, "this program requires exactly 2 processes.\n");
        }
//arreter proprement gracefully // finalize est pour arreter l environnement
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(EXIT_FAILURE);
    }
start_time = MPI_Wtime();
sleep(1);
end_time = MPI_Wtime();

printf("Process %d out of %d is running on %s\n", myrank, size, hostname);
printf("Execution time on process %d: %f seconds\n", myrank, end_time - start_time);

MPI_Finalize();
}
