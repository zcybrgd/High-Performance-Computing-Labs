#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  //ce programme a besoin de 4 processus pour fonctionner (un emetteur et 3 recepteurs)
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(size!=4){
    printf("Ce programme a besoin de 4 noeuds\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  switch (myrank) {
    case 0: {
      // Processus rang 0 envoie les messages.
      int buffer[3] = {123, 456, 789};
      MPI_Request requests[3];
      int recipient_rank_of_request[3];
      // Envoyer le premier message au processus 1
      printf("[Processus %d] Envoie %d au processus 1.\n", my_rank, buffer[0]);
      MPI_Issend(&buffer[0], 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &requests[0]);
      recipient_rank_of_request[0] = 1;
      // Envoyer le premier message au processus 2
      printf("[Processus %d] Envoie %d au processus 2.\n", my_rank, buffer[1]);
      MPI_Issend(&buffer[1], 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &requests[1]);
      recipient_rank_of_request[1] = 2;
      // Envoyer le premier message au processus 3
      printf("[Processus %d] Envoie %d au processus 3.\n", my_rank, buffer[2]);
      MPI_Issend(&buffer[2], 1, MPI_INT, 3, 0, MPI_COMM_WORLD, &requests[2]);
      recipient_rank_of_request[2] = 3;
      // Barrière pour s'assurer que les envois 1 et 2 sont complets au premier MPI_Waitsome
      MPI_Barrier(MPI_COMM_WORLD);
      // Attendez que l'un des envois non bloquantsse termine
      int index_count;
      int indices[3];
      MPI_Waitsome(3, requests, &index_count, indices, MPI_STATUSES_IGNORE);
      for(int i = 0; i < index_count; i++) {
        printf("[Processus %d] Premier MPI_Waitsome: l'envoi au processus non bloquant%d est complet.\n", my_rank, recipient_rank_of_request[indices[i]]);
      }
      // Barrière pour s'assurer que l'envoi 3 est terminé à la seconde MPI_Waitsome
      MPI_Barrier(MPI_COMM_WORLD);
      // Attendez que l'autre envoi non bloquantsoit terminé
      MPI_Waitsome(3, requests, &index_count, indices, MPI_STATUSES_IGNORE);
      for(int i = 0; i < index_count; i++) {
        printf("[Processus %d] Seconde MPI_Waitsome: l'envoi au processus non bloquant%d est complet.\n", my_rank, recipient_rank_of_request[indices[i]]);}
      break;
    }
    case 3: {
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      // Le dernier processus attendra sur la barrière avant de recevoir le message.
      int received;
      MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("[Processus %d] Reçu la valeur%d.\n", my_rank, received);
      break;
    }
    default: {
      // Les processus 1 et 2 recevront le message, puisils attendrontsur la barrière.
      int received;
      MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("[Processus %d] Reçu la valeur %d.\n", my_rank, received);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    }
  }
  
  MPI_Finalize();
  return 0;
}
