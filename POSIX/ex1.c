#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// les arguments to be sent inside when creating a new thread (we have to enter it as a datastructure)
typedef struct {
    int num1;
    int num2;
    int thread_id;
} ThreadArgs;

// the routine to be executed by the thread
void* calculate_sum(void* args) {
    ThreadArgs* arguments = (ThreadArgs*)args;

    //Enables or disables thread cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

    printf("Thread %d: Calculating the sum of %d and %d\n", arguments->thread_id, arguments->num1, arguments->num2);
    int sum = arguments->num1 + arguments->num2;

    // ???????????
    sleep(2);

    printf("Thread %d: The sum of %d and %d is %d\n", arguments->thread_id, arguments->num1, arguments->num2, sum);
    pthread_exit((void*)(intptr_t)sum); // Exit the thread, returning the sum
}
//intptr_t is a signed integer type capable of storing the value of a pointer. It is guaranteed to have the same size as a pointer on the platform.
int main() {
    int num_threads = 5;
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];
    pthread_attr_t attr;

    // ???????????
    pthread_attr_init(&attr);

    // ???????????
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);

    // ???????????
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // ???????????
    for (int i = 0; i < num_threads; i++) {
        args[i].num1 = i;
        args[i].num2 = i + 1;
        args[i].thread_id = i + 1;

        if (pthread_create(&threads[i], &attr, calculate_sum, (void*)&args[i])) {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    // ???????????
    sleep(1);
    printf("Main: Canceling thread 3\n");
    pthread_cancel(threads[2]); // ????????

    // ???????????
    for (int i = 0; i < num_threads; i++) {
        void* status;
        if (pthread_join(threads[i], &status)) {
            fprintf(stderr, "Error joining thread\n");
            return 2;
        }
        if (status == PTHREAD_CANCELED) {
            printf("Thread %d was canceled\n", i + 1);
        } else {
            printf("Thread %d exited with status %ld\n", i + 1, (intptr_t)status);
        }
    }

    // ???????????
    pthread_attr_destroy(&attr);
    return 0;
}
