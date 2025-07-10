#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

//config
#define DEFAULT_WIDTH   500     // Largeur par défaut - Default width
#define DEFAULT_HEIGHT  500     // Hauteur par défaut - Default height
#define DEFAULT_STEPS   100     // Nombre d'étapes par défaut - Default steps
#define FIXED_PROB     0.3     // Probabilité fixe de cellule vivante - Fixed alive cell probability
#define SEED           42       // Graine aléatoire fixe - Fixed random seed

// =======================
// Macro gestion d'erreur CUDA
// =======================
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error: %s (code %d), line %d, file %s\n",    \
                cudaGetErrorString(err), err, __LINE__, __FILE__);          \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)



struct Simulation {
    int width;      // Largeur de la grille - Grid width
    int height;     // Hauteur de la grille - Grid height
    int steps;      // Nombre d'itérations 
    int **current;  // Grille actuelle 
    int **next;     // Grille suivante 
};


__device__ int getGlobalIndex2D(int i, int j, int width) {
    return i * width + j;
}

//mesure du temps haute précision 
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

//allocation contiguë de la grille
int** allocate_grid(int width, int height) {
    int **grid = (int**)malloc(width * sizeof(int*));
    int *data = (int*)malloc(width * height * sizeof(int));
    for (int i = 0; i < width; i++) {
        grid[i] = &data[i * height];
    }
    return grid;
}

//init random
void init_grid(struct Simulation *sim) {
    srand(SEED);
    for (int i = 0; i < sim->width; i++) {
        for (int j = 0; j < sim->height; j++) {
            sim->current[i][j] = (rand()/(double)RAND_MAX) < FIXED_PROB ? 1 : 0;
        }
    }
}


__device__ int count_neighbors(int width,int height, int* current, int x, int y) {
    int count = 0;
    for (int di = -width-1; di <= width+1; di+=width+1 ) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;  //ignorer la cellule elle-même
            int ni = (x + di + width) % width; 
            int nj = (y + dj + height) % height;
            count += current[getGlobalIndex2D(ni, nj, width)];
        }
    }
    return count;
}


__global__ void update_grid(int* current, int* next, int width, int height) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < height && j < width) {
        int idx = getGlobalIndex2D(i, j, width);
        int alive = count_neighbors(width, height, current, i, j);
        // Règles du Jeu de la Vie - Game of Life rules:
        next[idx] = (alive == 3) || (current[idx] && alive == 2);
    }
}

//affichage (pour les petites grilles)
void print_grid(struct Simulation *sim) {
    if (sim->width > 50 || sim->height > 50) return;

    for (int j = 0; j < sim->height; j++) {
        for (int i = 0; i < sim->width; i++) {
            printf("%c", sim->current[i][j] ? '#' : '.');
        }
        printf("\n");
    }
}

//libération mémoire
void free_simulation(struct Simulation *sim,int *d_current, int *d_next) {
    free(sim->current[0]);
    free(sim->current);
    free(sim->next[0]);
    free(sim->next);
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
}


int main() {

    struct Simulation sim;

    //input of user
    printf("Entrez la largeur de la grille (current %d): ", DEFAULT_WIDTH);
    scanf("%d", &sim.width);
    printf("Entrez la hauteur de la grille (current %d): ", DEFAULT_HEIGHT);
    scanf("%d", &sim.height);
    printf("Entrez le nombre d'iterations (current %d): ", DEFAULT_STEPS);
    scanf("%d", &sim.steps);

    //Validation des entrées
    if (sim.width <= 0 || sim.height <= 0 || sim.steps <= 0) {
        printf("Erreur: Les dimensions doivent etre positives!\n");
        return 1;
    }

    size_t size = sim.width * sim.height * sizeof(int);

    //init
    sim.current = allocate_grid(sim.width, sim.height);
    sim.next = allocate_grid(sim.width, sim.height);
    init_grid(&sim);

    printf("\n=== Jeu de la Vie ===\n");
    printf("Taille: %dx%d\nIterations: %d\nProbabilite: %.2f\n", sim.width, sim.height, sim.steps, FIXED_PROB);


    //allocation sur GPU
    int* d_current = nullptr;
    int* d_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_current, size));
    CUDA_CHECK(cudaMalloc(&d_next, size));

    // copie vers GPU
    CUDA_CHECK(cudaMemcpy(d_current, sim.current, size, cudaMemcpyHostToDevice));

    // Définition des blocs/threads
    dim3 blockDim(16, 16);
    dim3 gridDim((sim.width + blockDim.x - 1) / blockDim.x,(sim.height + blockDim.y - 1) / blockDim.y);

    printf("Lancement kernel avec grid (%d, %d) blocs et blockDim (%d, %d) threads\n",gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Création évènements GPU pour mesurer le temps
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

      /* Simulation */
    for (int s = 0; s < sim.steps; s++) {
        // Lancement kernel
        update_grid<<<gridDim, blockDim>>>(d_current, d_next, sim.width, sim.height);
        //Swap grids
        CUDA_CHECK(cudaGetLastError());
        int* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
    // Check erreur lancement kernel
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Temps GPU kernel: %.3f ms\n", milliseconds);

    // Copie retour CPU
    CUDA_CHECK(cudaMemcpy(sim.current, d_current, size, cudaMemcpyDeviceToHost));

    /* Résultats */
    printf("\n=== Resultats ===\n");
    printf("Performance: %.2f cellules/seconde\n",
           (sim.width * sim.height * sim.steps) / (milliseconds*pow(10,-3)));

    if (sim.width <= 50 && sim.height <= 50) {
        printf("\nConfiguration finale:\n");
        print_grid(&sim);
    }

    free_simulation(&sim,d_current,d_next);

    // Destruction évènements
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
