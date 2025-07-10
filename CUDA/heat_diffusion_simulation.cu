/**
Le programme nécessite un seul argument en ligne de commande :

-> Taille de la grille (siz) : Un entier qui définit les dimensions de la grille (c'est-à-dire que la grille sera de taille siz x siz). 
Cette valeur détermine la résolution de la simulation et influence directement la complexité de calcul ; par exemple “./heat 100”.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <math.h>

__host__ __device__ double square(double x) {
    return x * x;
}

__global__ void updateInterior(double *currentGrid, double *newGrid, int gridSize, double diffusionCoeff, double dt, double dx) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx> 0 && idx < gridSize-1 && idy > 0 && idy < gridSize-1) {
    newGrid[idx*gridSize+idy] = currentGrid[idx*gridSize+idy] + diffusionCoeff * dt * (currentGrid[(idx-1)*gridSize+idy] + currentGrid[(idx+1)*gridSize+idy] + currentGrid[idx*gridSize+idy-1] + currentGrid[idx*gridSize+idy+1] - 4 * currentGrid[idx*gridSize+idy]) / square(dx);
    }
}

__global__ void updateBoundaries(double *currentGrid, double *newGrid, int gridSize, double diffusionCoeff, double dt, double dx) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y +1;


  if(idx==0 && idy<gridSize-1){
newGrid[idy] = currentGrid[idy] + diffusionCoeff * dt * (currentGrid[(gridSize-1)*gridSize+idy] + currentGrid[gridSize+idy] + currentGrid[idy - 1] + currentGrid[idy*gridSize + 1] - 4 * currentGrid[idy]) / square(dx);
  }

  if(idx==gridSize-1 && idy<gridSize-1){

newGrid[idx*gridSize+idy] = currentGrid[(gridSize - 1)*gridSize+idy] + diffusionCoeff * dt * (currentGrid[(gridSize - 2)*gridSize+idy] + currentGrid[idy] + currentGrid[(gridSize - 1)*gridSize + idy - 1] + currentGrid[(gridSize - 1)*gridSize+(idy + 1)] - 4 * currentGrid[(gridSize - 1)*gridSize+idy]) / square(dx);

   }

}

__global void calculerMoyenne(double *currentGrid, int gridSize, double averageTemp) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < gridSize && idy < gridSize) {
    atomicAdd(averageTemp, currentGrid[idx * gridSize + idy]);
    }

}
__global__ void initGrid(double *currentGrid, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < gridSize && idy < gridSize) {
        currentGrid[idx * gridSize + idy] = 0;
    }
}

__global__ void modif2extremites(double* currentGrid, int gridSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < gridSize && idy == 0) {
        currentGrid[idx * gridSize + idy] = square(cos(idx * M_PI / (double)gridSize));
    }
    if (idx < gridSize && idy == gridSize - 1) {
        currentGrid[idx * gridSize + idy] = square(sin(idx * M_PI / (double) gridSize));
    }

}

int main(int argc,char* argv[]){
  clock_t startTime;
  startTime = clock();
  const int gridSize = atoi(argv[1]);
  const double diffusionCoeff = 1;
  double averageTemp = 0;

  double *currentGrid, *newGrid;

  cudaMalloc(&(currentGrid),gridSize*gridSize*sizeof(double));
  cudaMalloc(&(newGrid),gridSize*gridSize*sizeof(double));


  dim3 blockDim(16, 16);
  dim3 gridDim((gridSize + blockDim.x - 1) / blockDim.x,
             (gridSize + blockDim.y - 1) / blockDim.y);


  double dx = M_PI / gridSize;
  const double dt = square(dx) / (8 * diffusionCoeff);
  const double totalTime = 0.5 * square(M_PI) / diffusionCoeff;
  const double numSteps = totalTime / dt;

  //initialisation de la grille
  initGrid<<<1, 1>>>(currentGrid, gridSize);

  modif2extremites<<<1,1>>>(currentGrid, gridSize);

  printf("Size is %d\n", gridSize);

  //to copy the full grid
  cudaMemcpy(newGrid, currentGrid, gridSize * gridSize * sizeof(double), cudaMemcpyDeviceToDevice);

   //boucle pour inside la grille
  for (int step = 0; step < numSteps; step++) {

        updateInterior<<<1, 1>>>(currentGrid, newGrid, gridSize, diffusionCoeff, dt, dx);
        cudaDeviceSynchronize();

        updateBoundaries<<<1, 1>>>(currentGrid, newGrid, gridSize, diffusionCoeff, dt, dx);
        cudaDeviceSynchronize();

        //refill the current grid with new values
        cudaMemcpy(currentGrid, newGrid, gridSize * gridSize * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    //reduction pour calculer la moyenne
    double *d_averageTemp;
cudaMalloc(&d_averageTemp, sizeof(double));
cudaMemset(d_averageTemp, 0, sizeof(double));
calculerMoyenne<<<1, 1>>>(currentGrid, gridSize, d_averageTemp);
cudaMemcpy(&averageTemp, d_averageTemp, sizeof(double), cudaMemcpyDeviceToHost);
cudaFree(d_averageTemp);


    averageTemp /= gridSize * gridSize;
    printf("Average temperature is %f\n", averageTemp);
    clock_t endTime;
    endTime = clock();
    double elapsedTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
    printf("Execution time is %f seconds\n", elapsedTime);
    for(int i=0;i<gridSize;i++){
    cudaFree(currentGrid[i]);
    cudaFree(newGrid[i]);
  }
  cudaFree(currentGrid);
  cudaFree(newGrid);

  return 0;

}
