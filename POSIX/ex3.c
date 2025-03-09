#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define LOOP_COUNT 10000000

void* increment(void* param);

unsigned int counter=0;
pthread_mutex_t mutex;

int main() {
    pthread_t thread1, thread2;

    pthread_mutex_init(&mutex, NULL); 

    pthread_create(&thread1, NULL, increment, NULL);
    pthread_create(&thread2, NULL, increment, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&mutex);
    printf("counter:%d\n", counter);
    exit(0);
}

void* increment(void* param) {

    int j = 0;

    for (; j < LOOP_COUNT; j++) {
        pthread_mutex_lock(&mutex);
        counter++;
        pthread_mutex_unlock(&mutex);
    }

    return NULL;
}
