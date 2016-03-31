#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int cmp_func(const void *a , const void *b)
{
    return ( *(int *) a - *(int *) b );
}

int difference(int a[], int b[], int n){
    
    int i;
    for(i = 0 ; i < n ; ++i){
        if(a[i] != b[i])
            return 0;
    }
    return 1;
}


int main(int argc, char* argv[])
{
  int i, j;
  int n, count;
  int numberOfThreads;
  double t_start, t_end, t_count_sort, t_quicksort;
  
  srand(time(NULL));
  
  // Get array size and number of threads from the command line                                       
  if (argc > 1) {
      n = strtol(argv[1], NULL, 10);
      numberOfThreads = strtol(argv[2], NULL, 10);
  }
  else {
      printf("\nWhen running this program, please include the array size and the number of threads on command line.\n");
      return 0;
  }

  n = atoi(argv[1]);
  fprintf(stdout, "n = %d\n", n);
  
  int *a = malloc(n*sizeof(int));
  int *b = malloc(n*sizeof(int));
  int *temp = malloc(n*sizeof(int));
  
  for (i = 0; i < n; ++i) {
    b[i] = a[i] = rand() % 1000;
    temp[i] = 0;
  }  
    

    // count_sort starts here

    t_start = omp_get_wtime();
    
    // TODO: - Parallelize the i-loop and j-loop
    //       - Also implement versions with offloads to Xeon Phis
    //       - You will turn in 4 different source code files for this problem:
    //         1. i-loop cpu, 2. j-loop cpu, 3. i-loop phi, 4. j-loop phi
#pragma offload target(mic) in(n) inout(count) inout(a,temp:length(n))
{
#pragma omp parallel for num_threads(numberOfThreads) shared(a,n,temp) private(count,i,j)
    for(i = 0 ; i < n ; i++){
        count = 0;
        for(j = 0 ; j < n ; j++){
            if(a[j] < a[i])
                count++;
            else if((a[j] == a[i]) && (j < i))
                count++;
        }
        
        temp[count] = a[i];
    }

#pragma omp barrier
    
    // TODO: Modify the code below so that the copy can be made OpenMP parallel
#pragma omp parallel for shared(a,n,temp)
	for(i=0;i<n;i++)
	a[i]=temp[i];
}
    t_end = omp_get_wtime();
    
    t_count_sort = t_end - t_start;
    
    printf("Time needed for count sort using %d threads = %lf\n",
           numberOfThreads, t_end - t_start);
    
    //count_sort ends here
    
    
    //quicksort starts
    
    t_start = omp_get_wtime();
    
    qsort(b, n, sizeof(int), cmp_func);
    
    t_end = omp_get_wtime();
    
    t_quicksort = t_end - t_start ;
    
    printf("time needed for sequential Quicksort = %lf\n", t_quicksort);
    
    
    //compare the results
    
    if (difference(a,b,n) == 0)
        printf("Wrong Ans\n");
    else printf("Correct Ans\n");
    
    free(a);
    free(b);
    free(temp);

    
  return 0;
}
