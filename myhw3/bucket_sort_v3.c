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
#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)

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
    int N,P,my_rank,n,S,s;
    int i,j,k;
    int counter;
    double temp;
    double *rawA, *sendA, *recvA, *sortA;
    double *sample, *root_sample, *pivot;
    int *scounter, *sdispl, *rcounter, *rdispl, *root_counter, *root_displ;

    double t0,t1,t2,t3,t4,t5,t6,t7,t8,t_start,t_end,t_tot,t_max,t_min,t_avg;
    double *t;

    if(argc>1){
        N=strtol(argv[1],NULL,10);
    }
    else{
        printf("Please include the array size on the command line.\n");
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    if(my_rank==0){
        t0=MPI_Wtime();
    }

    n=N/P;
    S=12*P*log10(N);
    s=12*log10(N);

    rawA=(double*)malloc(n*sizeof(double));
    sendA=(double*)malloc(n*sizeof(double));
    recvA=(double*)malloc(N*sizeof(double));

    scounter=(int*)malloc(P*sizeof(int));
    sdispl=(int*)malloc(P*sizeof(int));
    rcounter=(int*)malloc(P*sizeof(int));
    rdispl=(int*)malloc(P*sizeof(int));

    sample=(double*)malloc(s*sizeof(double));
    pivot=(double*)malloc((P+1)*sizeof(double));

    if(my_rank==0){
        sortA=(double*)malloc(N*sizeof(double));
        root_counter=(int*)malloc(P*sizeof(int));
        root_displ=(int*)malloc(P*sizeof(int));
        root_sample=(double*)malloc(S*sizeof(double));

        t=(double*)malloc(P*sizeof(double));
    }

    srand(my_rank+time(NULL));

    if(my_rank==0){
        t1=MPI_Wtime();
    }

//    printf("raw data from rank %d", my_rank);
    for(i=0;i<n;i++){
        temp=((double)rand()/RAND_MAX);
        rawA[i]=temp*temp;
//        printf("%f  ",rawA[i]);
        if(i<s) sample[i]=rawA[i];
    }
//    printf("\n");

    if(my_rank==0){
        t2=MPI_Wtime();
        printf("Time for generate: %.2f seconds.\n",t2-t1);
    }

    MPI_Gather(sample,s,MPI_DOUBLE,root_sample,s,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);
    if(my_rank==0){
        qsort_dbls(root_sample,S);
        pivot[0]=0.0;
        pivot[P]=1.0;
        for(i=1;i<P;i++){
            pivot[i]=root_sample[S*i/P];
        }
    }

    MPI_Bcast(pivot,P+1,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        t3=MPI_Wtime();
        printf("Time for pivot: %.2f seconds.\n",t3-t2);
    }

    k=0;
    for(j=0;j<P;j++){
        scounter[j]=0;
        for(i=0;i<n;i++){
            if(rawA[i]>=pivot[j] && rawA[i]<pivot[j+1]){
                sendA[k]=rawA[i];
                scounter[j]++;
                k++;
            }
        }
    }

    sdispl[0]=0;
    for(j=0;j<P-1;j++)
        sdispl[j+1]=sdispl[j]+scounter[j];

    if(my_rank==0){
        t4=MPI_Wtime();
        printf("Time for bin: %.2f seconds.\n",t4-t3);
    }

    MPI_Alltoall(scounter,1,MPI_INT,rcounter,1,MPI_INT,MPI_COMM_WORLD);

    rdispl[0]=0;
    for(j=0;j<P-1;j++)
        rdispl[j+1]=rdispl[j]+rcounter[j];

    MPI_Alltoallv(sendA,scounter,sdispl,MPI_DOUBLE,recvA,rcounter,rdispl,MPI_DOUBLE,MPI_COMM_WORLD);

    counter=rdispl[P-1]+rcounter[P-1];

    if(my_rank==0){
        t5=MPI_Wtime();
        printf("Time for distribute: %.2f seconds.\n",t5-t4);
    }

    t_start=MPI_Wtime();

    qsort_dbls(recvA,counter);

    t_end=MPI_Wtime();

    if(my_rank==0){
        t6=MPI_Wtime();
        printf("Time for local sort: %.2f seconds.\n",t6-t5);
    }

    MPI_Gather(&counter,1,MPI_INT,root_counter,1,MPI_INT,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        root_displ[0]=0;
        for(j=0;j<P-1;j++){
            root_displ[j+1]=root_displ[j]+root_counter[j];
        }
    }

    MPI_Gatherv(recvA,counter,MPI_DOUBLE,sortA,root_counter,root_displ,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        t7=MPI_Wtime();
        printf("Time for gather: %.2f seconds.\n",t7-t6);
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

    free(rawA);
    free(sendA);
    free(recvA);
    free(scounter);
    free(sdispl);
    free(rcounter);
    free(rdispl);
    free(sample);
    free(pivot);

    if(my_rank==0){
        free(sortA);
        free(root_counter);
        free(root_displ);
        free(root_sample);
    }

    if(my_rank==0){
        t8=MPI_Wtime();
        printf("Total time(N=%d,P=%d): %.2f seconds.\n",N,P,t8-t0);
    }

    t_tot=t_end-t_start;
    MPI_Gather(&t_tot,1,MPI_DOUBLE,t,1,MPI_DOUBLE,mpi_root,MPI_COMM_WORLD);

    if(my_rank==0){
        t_max=t[0];
        t_min=t[0];
        t_avg=t[0];
        for(i=1;i<P;i++){
            t_max=MAX(t_max,t[i]);
            t_min=MIN(t_min,t[i]);
            t_avg=t_avg+t[i];
        }

        printf("Max time: %.2f\n", t_max);
        printf("Min time: %.2f\n", t_min);
        printf("Avg time: %.2f\n", t_avg/P);

        free(t);
    }

    MPI_Finalize();

    return 0;

}
