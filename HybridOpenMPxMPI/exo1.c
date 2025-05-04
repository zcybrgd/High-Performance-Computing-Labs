#include <mpi.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int thread_id, num_threads;
  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  #pragma omp parallel private(thread_id)
  {
    thread_id = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("\nHello world from thread %d of %d \nfrom machine %d in %d the processors name : %s\n",thread_id,num_threads,world_rank,world_size,processor_name);
  }
  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}
