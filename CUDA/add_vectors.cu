#include <stdlib.h>
#include <stdio.h>


#define N 512

__global__ void add ( int *a , int *b , int * c ) {
c [ blockIdx.x ] = a [ blockIdx.x ] + b [ blockIdx.x ] ;
}

int main( void ) {
int *a, *b, *c; //copies de a, b, c de Host
int
*dev_a, *dev_b, *dev_c; //copies de a, b, c de Device
int size = N * sizeof( int);
// allocation de l’espace pour le device
cudaMalloc( (void**)&dev_a, size);
cudaMalloc( (void**)&dev_b, size);
cudaMalloc( (void**)&dev_c, size);

a = (int*)malloc( size );
b = (int*)malloc( size );
c = (int*)malloc( size );
random_ints( a, N );
random_ints( b, N );

// Copie des données vers le Device
cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice);

// Lancer kernel add() kernel avec N parallèles blocs
add<<< N, 1 >>>( dev_a, dev_b, devc);

//Copie du resultat vers Host
cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost);

//Liberation de l’espace alloué
free( a ); free( b ); free( c):
cudaFree( dev_a );
cudaFree( dev_b );
cudaFree( dev_c );
return 0; }
