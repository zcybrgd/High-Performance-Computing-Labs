# High-Performance-Computing-Labs

sharing my practical implementations and experiments in High-Performance Computing lab sessions (HPC). 

it includes exercises, demonstrations, and comparisons of different parallelization strategies to build a solid understanding of HPC fundamentals from the HPC course we take in second semester of our first masters year in ESI Alger.

## Content

- `CUDA/`: CUDA programs leveraging GPU parallelism.
- `POSIX/`: Programs using Pthreads to implement classic synchronization and parallel computation patterns.
- `OpenMP/`: Examples with OpenMP directives for shared-memory multi-threading.
- `MPI/`: Programs using MPI to distribute computations across multiple processes, typically on multiple nodes of a cluster (we used a cluster of 2 machines during the labs).
- `HybridOpenMPXMPI/`: Hybrid implementations combining MPI with OpenMP to combine thread-level and process-level parallelism. (wanted to add with POSIX too but havent got the time)
