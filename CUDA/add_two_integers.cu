#include <stdio.h>
__global__ void add ( int *a , int*b , int *c ) {
*c = *a + *b ;
}
int main ( void ) {
int a , b , c ; //copies de a, b, c de Host
int *dev_a , *dev_b , *dev_c ; //pointeurs vers des zones dans le GPU (device)
int size = sizeof ( int ) ;
// allocation de l’espace pour le device
cudaMalloc( (void **)&dev_a, size) ;
cudaMalloc( (void **)&dev_b, size) ;
cudaMalloc( (void **)&dev_c, size) ;
a=2 ;
b=7 ;
// Copie des données vers le Device
cudaMemcpy( dev_a , &a , size , cudaMemcpyHostToDevice );
cudaMemcpy( dev_b , &b , size , cudaMemcpyHostToDevice);
add <<< 1 , 1 >>> ( dev_a , dev_b , dev_c ) ;
//Copie du resultat vers Host
cudaMemcpy (&c, dev_c, size, cudaMemcpyDeviceToHost) ;
//Liberation de l’espace alloué
cudaFree (dev_a) ;
cudaFree ( dev_b) ;
cudaFree ( dev_c) ;
return 0
}


/*
nvcc -o add_two_integers add_two_integers.cu
./add_two_integers

*/
