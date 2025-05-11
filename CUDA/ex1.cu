%%writefile ex1.cu
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>

/*

plus de blocs is better psk si un groupe de thread en divergence on peut pas laisser les cores en attente pour masquer un petit peu le GPU
*/

  
// Le nombre de threads par blocs dans le kernel

const int threads_per_block = 256;


// Déclarations
float *GPU_add_vectors(float *A, float *B, int N);
float *CPU_add_vectors(float *A, float *B, int N);
float *get_random_vector(int N);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void die(const char *message);


int main(int argc, char **argv) {
	// Amorcer le générateur aléatoire (utiliser une constante ici pour des résultats reproductibles)
	srand(10);

	// Déterminer la taille du vecteur
	int N = 100000000;
	if (argc > 1) N = atoi(argv[1]); // valeur spécifiée par l'utilisateur

	// Générer deux vecteurs aléatoires
	long long vector_start_time = start_timer();
	float *A = get_random_vector(N);
	float *B = get_random_vector(N);
	stop_timer(vector_start_time, "Vector generation");

	// La somme des deux vecteurs sur GPU
	long long GPU_start_time = start_timer();
	float *C_GPU = GPU_add_vectors(A, B, N);
	long long GPU_time = stop_timer(GPU_start_time, "\t            Total");

	// La somme des deux vecteurs sur CPU
	long long CPU_start_time = start_timer();
	float *C_CPU = CPU_add_vectors(A, B, N);
	long long CPU_time = stop_timer(CPU_start_time, "\nCPU");

	// Calculer l'accélération ou la dégradation
	if (GPU_time > CPU_time) printf("\nCPU outperformed GPU by %.2fx\n", (float) GPU_time / (float) CPU_time);
	else                     printf("\nGPU outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);

	// Vérifier l'exactitude des résultats du GPU
	int num_wrong = 0;
	for (int i = 0; i < N; i++) {
		if (fabs(C_CPU[i] - C_GPU[i]) > 0.000001) num_wrong++;
	}

	// Rapporter les résultats d'exactitude
	if (num_wrong) printf("\n%d / %d values incorrect\n", num_wrong, N);
	else           printf("\nAll values correct\n");

}


// Le kernel du GPU qui calcule la somme  A + B
// (chaque thread calcule une seule valeur du résultat)
__global__ void add_vectors_kernel(float *A, float *B, float *C, int N) {
	// Déterminez quel élément ce thread calcule
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	int thread_id = blockDim.x * block_id + threadIdx.x;

	// Calculer un seul élément du vecteur de résultat (si l'élément est valide)
	if (thread_id < N) C[thread_id] = A[thread_id] + B[thread_id];
}


// Retourner  la somme  A + B (calculée sur le GPU)
float *GPU_add_vectors(float *A_CPU, float *B_CPU, int N) {

	long long memory_start_time = start_timer();

	// Allouer de la mémoire GPU pour les entrées et le résultat
	int vector_size = N * sizeof(float);
	float *A_GPU, *B_GPU, *C_GPU;
	if (cudaMalloc((void **) &A_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &B_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");
	if (cudaMalloc((void **) &C_GPU, vector_size) != cudaSuccess) die("Error allocating GPU memory");

	// Transférer les vecteurs d'entrée vers la mémoire GPU
	cudaMemcpy(A_GPU, A_CPU, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, vector_size, cudaMemcpyHostToDevice);

	stop_timer(memory_start_time, "\nGPU:\t  Transfer to GPU");

	// Déterminer le nombre de blocs dans les dimensions x et y
	int num_blocks = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block);
	int max_blocks_per_dimension = 65535;
  //since the number of blocks per dimension is limited. To fit within this constraint, we compute the number of blocks along the y-axis
	int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
	int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);

	// Exécuter le kernel pour calculer la somme  sur le GPU
	long long kernel_start_time = start_timer();
	add_vectors_kernel <<< grid_size , threads_per_block >>> (A_GPU, B_GPU, C_GPU, N);
	cudaDeviceSynchronize(); // this is only needed for timing purposes
	stop_timer(kernel_start_time, "\t Kernel execution");

	// Rechercher les erreurs du kernel
	cudaError_t error = cudaGetLastError();
	if (error) {
		char message[256];
		sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
		die(message);
	}

	// Allouer de la mémoire CPU pour le résultat
	float *C_CPU = (float *) malloc(vector_size);
	if (C_CPU == NULL) die("Error allocating CPU memory");

	// Transférer le résultat du GPU vers le CPU
	memory_start_time = start_timer();
	cudaMemcpy(C_CPU, C_GPU, vector_size, cudaMemcpyDeviceToHost);
	stop_timer(memory_start_time, "\tTransfer from GPU");

	// Libérer la mémoire du GPU
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);

	return C_CPU;
}


// Retourner la somme A + B
float *CPU_add_vectors(float *A, float *B, int N) {
	// Allouer de la mémoire pour le résultat
	float *C = (float *) malloc(N * sizeof(float));
	if (C == NULL) die("Error allocating CPU memory");

	// Calculer la somme;
	for (int i = 0; i < N; i++) C[i] = A[i] + B[i];

	// Retourner le résultat
	return C;
}


// Retourner un vecteur aléatoire contenant N éléments
float *get_random_vector(int N) {
	if (N < 1) die("Number of elements must be greater than zero");

	// Allouer de la mémoire pour le vecteur
	float *V = (float *) malloc(N * sizeof(float));
	if (V == NULL) die("Error allocating CPU memory");

	// Remplir le vecteur avec des nombres aléatoires
	for (int i = 0; i < N; i++) V[i] = (float) rand() / (float) rand();


	return V;
}


// Retourner le temps
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Retourner le temps écoulé depuis le temps spécifié spécifié
long long stop_timer(long long start_time, const char *name) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));
	return end_time - start_time;
}


// exit
void die(const char *message) {
	printf("%s\n", message);
	exit(1);
}

/*
Equation 1: Calculating num_blocks
We need to determine how many blocks are required to process N elements, given that each block contains threads_per_block threads. Since N might not be an exact multiple of threads_per_block, we need to round up:

num_blocks = ⌈ 𝑁 /threads_per_block ⌉
Since integer division in C/C++ truncates toward zero, we rewrite it in a way that ensures correct rounding:

num_blocks = 𝑁 + threads_per_block − 1 / threads_per_block
This formula ensures that if N isn't an exact multiple of threads_per_block, an extra block is allocated.

Equation 2: Determining Grid Dimensions (num_blocks_x and num_blocks_y)
CUDA limits the number of blocks per dimension (max_blocks_per_dimension = 65535). To ensure we don’t exceed this, we distribute the blocks across x and y dimensions.

First, we calculate the number of blocks required in the y-dimension:

num_blocks_y = ⌈ num_blocks / max_blocks_per_dimension ⌉
Rewritten using integer math:

num_blocks_y = num_blocks + max_blocks_per_dimension − 1 / max_blocks_per_dimension

Now, we distribute the remaining blocks across the x-dimension, ensuring num_blocks_x * num_blocks_y covers all num_blocks:

num_blocks_x = num_blocks + num_blocks_y − 1 /num_blocks_y

Why Do We Do This?
The first equation ensures efficient distribution of threads over blocks.

The second equation ensures we respect CUDA’s limitations while distributing blocks across dimensions for optimal performance.

*/
