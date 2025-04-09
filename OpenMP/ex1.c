#include <stdio.h>
#include <omp.h>
#define SIZE  100
#define CHUNK  10

int main() {
  int tid;
  double a[SIZE], b[SIZE], c[SIZE];
  double start_time, end_time;
  for (size_t i = 0; i < SIZE; i++)
    a[i] = b[i] = i;
 start_time = omp_get_wtime();
  #pragma omp parallel private(tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0)
      printf("Nb threads = %d\n", omp_get_num_threads());
    printf("Thread %d: starting...\n", tid);

    //#pragma omp for schedule(dynamic, CHUNK)
   // #pragma omp for schedule(static)
    #pragma omp for schedule(static, CHUNK)
    for (size_t i = 0; i < SIZE; i++) {
      c[i] = a[i] + b[i];
      printf("Thread %d: c[%2zu] = %g\n", tid, i, c[i]);
    }
  }
   end_time = omp_get_wtime();

  printf("Time taken: %f seconds\n", end_time - start_time);
  return 0;
}
