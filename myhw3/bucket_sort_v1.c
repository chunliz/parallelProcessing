/*
CSE491-section2: parallel processing
Homework 3
Author: Chunli Zhang
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define mpi_root 0

int compare_dbls(const void* arg1, const void* arg2){
    double a1=*(double*) arg1;
    double a2=*(double*) arg2;
    if(a1<a2) return -1;
    else if(a1==a2) return 0;
    else return 1;
}

void qsort_dbls(double *array, int array_len){
    qsort(array, (size_t)array_len, sizeof(double), compare_dbls);
}

int main(int argc, char *argv[]){

    // define an array, which is initialized with random number
    int N,P,my_rank,n;
    int i,j,k;
    double *rawA,*sendA,*sortA,*my_A;
    double t0,t1,t2,t3,t4,t5,t6,t7;

    int counter;
    int *root_counter, *displs;
    MPI_Status status1,status2;

    if(argc>1){
        N=strtol(argv[1],NULL,10);
    }
    else{
        printf("Please include the array size on the command line.\n");
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    if(my_rank==0) t0=MPI_Wtime();

    my_A=(double*)malloc(N*sizeof(double));

//    printf("flag1 from rank %d\n",my_rank);
    if(my_rank==0){

        rawA=(double*)malloc(N*sizeof(double));
        sendA=(double*)malloc(N*sizeof(double));
        sortA=(double*)malloc(N*sizeof(double));
        root_counter=(int*)malloc(P*sizeof(int));
        displs=(int*)malloc(P*sizeof(int));

        t1=MPI_Wtime();

//        printf("raw:");
        for(i=0;i<N;i++){
            rawA[i]=((double)rand()/RAND_MAX);
//            printf("%f  ",rawA[i]);
        }
//        printf("\n");

        t2=MPI_Wtime();
        printf("Time for generate: %.2f seconds.\n",t2-t1);

        k=0;
        for(j=0;j<P;j++){
            root_counter[j]=0;

            for(i=0;i<N;i++){
                if(rawA[i]>=(double)j*1.0/(double)P && rawA[i]<((double)j+1)*1.0/(double)P){
                    sendA[k]=rawA[i];
		    root_counter[j]++;
                    k++;
                }
            }
        }

        displs[0]=0;
        for(j=0;j<P-1;j++){
            displs[j+1]=displs[j]+root_counter[j];
        }

        t3=MPI_Wtime();
        printf("Time for bin: %.2f seconds.\n",t3-t2);
    }

    MPI_Scatter(root_counter,1,MPI_INT,&counter,1,MPI_INT,mpi_root,MPI_COMM_WORLD);

    MPI_Scatterv(sendA,root_counter,displs,MPI_DOUBLE,my_A,counter,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        t4=MPI_Wtime();
        printf("Time for distribute: %.2f seconds.\n",t4-t3);
    }

/*
    if(my_rank==0){
       printf("flag2 from rank 0: %d\n",counter);

       for(i=0;i<counter;i++)
        printf("%f ",my_A[i]);
        printf("\n");
    }

    if(my_rank==1){
       printf("flag2 from rank 1: %d\n",counter);

       for(i=0;i<counter;i++)
        printf("%f ",my_A[i]);
        printf("\n");
    }
*/

    qsort_dbls(my_A,counter);

    if(my_rank==0){
        t5=MPI_Wtime();
        printf("Time for local sort: %.2f seconds.\n",t5-t4);
    }

    MPI_Gatherv(my_A,counter,MPI_DOUBLE,sortA,root_counter,displs,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        t6=MPI_Wtime();
        printf("Time for gather: %.2f seconds.\n",t6-t5);
    }

    if(my_rank==0){

        for(i=0;i<N-1;i++){
            if(sortA[i]>sortA[i+1]){
                printf("Wrong results!\n");
                break;
            }
        }

        printf("Correct results!\n");

//        printf("sorted:");
//        for(i=0;i<N;i++)
//            printf("%f  ",sortA[i]);
    }
//    printf("\n");

    if(my_rank==0){
        free(rawA);
        free(sendA);
        free(sortA);
        free(root_counter);
        free(displs);
    }

    free(my_A);

    if(my_rank==0){
        t7=MPI_Wtime();
        printf("Total time(N=%d,P=%d): %.2f seconds.\n",N,P,t7-t0);
    }

    MPI_Finalize();

    return 0;

}
