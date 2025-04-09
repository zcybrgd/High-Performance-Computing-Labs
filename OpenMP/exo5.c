#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#define WIDTH 256 // Image width
#define HEIGHT 256// Image height
// les sol possible:
// for collapse
// send the task (divide image in multiple region) 
// Function prototypes
void applyBlurFilter(int image[HEIGHT][WIDTH]);
void applySharpenFilter(int image[HEIGHT][WIDTH]);
void applyEdgeDetectionFilter(int image[HEIGHT][WIDTH]);
void initializeImage(int image[HEIGHT][WIDTH]);
void printImage(int image[HEIGHT][WIDTH]);

int main() {
     struct timeval start_program, end_program;
    gettimeofday(&start_program, NULL); // Start timing the entire program

    int image[HEIGHT][WIDTH];
    initializeImage(image);

    printf("Original Image:\n");
    // printImage(image);

    struct timeval start, end;
    gettimeofday(&start, NULL); // Start timing the filters

    applyBlurFilter(image);
    applySharpenFilter(image);
    applyEdgeDetectionFilter(image);
    
    gettimeofday(&end, NULL); // End timing the filters
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double cpu_time_used_filters = seconds + microseconds*1e-6;

    printf("Time taken (Sequential - Filters): %f seconds\n", cpu_time_used_filters);

    gettimeofday(&end_program, NULL); // End timing the entire program
    seconds = end_program.tv_sec - start_program.tv_sec;
    microseconds = end_program.tv_usec - start_program.tv_usec;
    double cpu_time_used_program = seconds + microseconds*1e-6;

    printf("Total Time taken (Entire Program): %f seconds\n", cpu_time_used_program);

    return 0;
}


void initializeImage(int image[HEIGHT][WIDTH]) {
    srand(time(NULL));  // Seed for random number generation
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            image[y][x] = rand() % 256;  // Random grayscale value
        }
    }
}

void applyBlurFilter(int image[HEIGHT][WIDTH]) {
    int tempImage[HEIGHT][WIDTH];
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            tempImage[y][x] = (image[y - 1][x] + image[y][x - 1] + image[y][x] + image[y][x + 1] + image[y + 1][x]) / 5;
        }
    }
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            image[y][x] = tempImage[y][x];
        }
    }
}

void applySharpenFilter(int image[HEIGHT][WIDTH]) {
    int tempImage[HEIGHT][WIDTH];
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            tempImage[y][x] = 5 * image[y][x] - (image[y - 1][x] + image[y][x - 1] + image[y][x + 1] + image[y + 1][x]);
            tempImage[y][x] = tempImage[y][x] < 0 ? 0 : (tempImage[y][x] > 255 ? 255 : tempImage[y][x]);
        }
    }
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            image[y][x] = tempImage[y][x];
        }
    }
}

void applyEdgeDetectionFilter(int image[HEIGHT][WIDTH]) {
    int tempImage[HEIGHT][WIDTH] = {0};
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int edge = image[y - 1][x] + image[y][x - 1] - 4 * image[y][x] + image[y][x + 1] + image[y + 1][x];
            tempImage[y][x] = edge < 0 ? 0 : (edge > 255 ? 255 : edge);
        }
    }
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            image[y][x] = tempImage[y][x];
        }
    }
}

void printImage(int image[HEIGHT][WIDTH]) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%3d ", image[y][x]);
        }
    }
    printf("\n");
}
