/*
Problème :
Deux ouvriers doivent accomplir une tâche nécessitant l'utilisation de deux outils. 
Chaque ouvrier tente de verrouiller un outil avant de verrouiller l'autre, ce qui peut entraîner un interblocage si les deux verrouillent d'abord un outil différent.

Problem:
Two workers need to complete a task requiring two tools. 
Each worker tries to lock one tool before locking the other, which may lead to a deadlock if both lock different tools first.
*/

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

void *Worker1(void *arg);
void *Worker2(void *arg);

pthread_mutex_t toolA = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t toolB = PTHREAD_MUTEX_INITIALIZER;

int main() {
    pthread_t tid1, tid2;
    pthread_create(&tid1, NULL, Worker1, NULL);
    pthread_create(&tid2, NULL, Worker2, NULL);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    return 0;
}

void *Worker1(void *arg) {
    pthread_mutex_lock(&toolA);
    sleep(10);
    pthread_mutex_lock(&toolB);
    printf("Worker1 has collected all the tools needed, they are starting the task!\n");
    pthread_mutex_unlock(&toolB);
    pthread_mutex_unlock(&toolA);
}

void *Worker2(void *arg) {
    pthread_mutex_lock(&toolB);
    sleep(10);
    pthread_mutex_lock(&toolA);
    printf("Worker2 has collected all the tools needed, they are starting the task!\n");
    pthread_mutex_unlock(&toolA);
    pthread_mutex_unlock(&toolB);
}
