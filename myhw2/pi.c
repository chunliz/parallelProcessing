#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double sum = 0.0;

// Method declarations
void* calculatePartialSum (void* threadNumber);
double sequentialCompute (long iterations);
double parallelCompute_atomic(long iterations, int numberOfThreads);
double parallelCompute_reduction(long iterations, int numberOfThreads);
double getDifference(double calculatedPi);

// Main method
int main(int argc, char* argv[])
{
  // Variable declarations
  double sequentialStart, sequentialEnd, sequentialTime;
  double parallelStart, parallelEnd, parallelTime_atomic, parallelTime_reduction;

  double sequentialPi, parallelPi_atomic, parallelPi_reduction;
  double sequentialDifference, parallelDifference_atomic, parallelDifference_reduction;
  long iterations; 
  int numberOfThreads;

  // Get number of iterations and number of threads from the command line
  if(argc > 1)
    {
      iterations = strtol(argv[1], NULL, 10);
      numberOfThreads = strtol(argv[2], NULL, 10);
    }
  else
    {
      printf("\nWhen running this program, please include number of iterations and number of threads on command line.\n");
      return 0;
    }

  // Time sequential calculation
  sequentialStart = omp_get_wtime();
  sequentialPi = sequentialCompute(iterations);
  sequentialEnd = omp_get_wtime();
  sequentialTime = sequentialEnd - sequentialStart;
  
  // Time parallel calculation with atomics
  parallelStart = omp_get_wtime();
  parallelPi_atomic = parallelCompute_atomic(iterations, numberOfThreads);
  parallelEnd = omp_get_wtime();
  parallelTime_atomic = parallelEnd - parallelStart;

  // Time parallel calculation with reduction
  parallelStart = omp_get_wtime();
  parallelPi_reduction = parallelCompute_reduction(iterations, numberOfThreads);
  parallelEnd = omp_get_wtime();
  parallelTime_reduction = parallelEnd - parallelStart;
  
  // How do results compare with PI?
  sequentialDifference = getDifference(sequentialPi);
  parallelDifference_atomic = getDifference(parallelPi_atomic);
  parallelDifference_reduction = getDifference(parallelPi_reduction);
  
  // Print results
  printf("Sequential Calculation: %f\n", sequentialPi);
  printf("This is %f away from the correct value of PI.\n\n", sequentialDifference);
  printf("ParallelCalculation with atomics: %f\n", parallelPi_atomic);
  printf("This is %f away from the correct value of PI.\n", parallelDifference_atomic);
  printf("Number of iterations: %ld, Number of Threads: %d\n", iterations, numberOfThreads);
  
  // Calculate the validity of the parallel computation
  double difference = parallelDifference_atomic - sequentialDifference;

  // if (difference < .01 && difference > -.01)
  if (parallelDifference_atomic < .01 && parallelDifference_atomic > -.01)
    printf("Parallel atomic calculation is VALID!\n");
  else
    printf("Parallel atomic calculation is INVALID!\n");

  // Calculate and print speedup results
  double speedup = ((double)sequentialTime)/((double)parallelTime_atomic);
  printf("Sequential Time: %f, Parallel Atomic Time: %f, Speedup: %f\n", sequentialTime, parallelTime_atomic, speedup);


  printf("\n\nParallelCalculation with reductions: %f\n", parallelPi_reduction);
  printf("This is %f away from the correct value of PI.\n", parallelDifference_reduction);
  printf("Number of iterations: %ld, Number of Threads: %d\n", iterations, numberOfThreads);
  
  // Calculate the validity of the parallel computation
  difference = parallelDifference_reduction - sequentialDifference;

  //if (difference < .01 && difference > -.01)
  if (parallelDifference_reduction < .01 && parallelDifference_reduction > -.01)
    printf("Parallel reduction calculation is VALID!\n");
  else
    printf("Parallel reduction calculation is INVALID!\n");

  // Calculate and print speedup results
  speedup = ((double)sequentialTime)/((double)parallelTime_reduction);
  printf("Sequential Time: %f, Parallel Reduction Time: %f, Speedup: %f\n", sequentialTime, parallelTime_reduction, speedup);

  return 0;
}


// TODO: You need to implement a sequential estimation for PI using the Monte Carlo method.
//       Use rand_r() here as well for fair comparison to OpenMP parallel versions.
double sequentialCompute (long iterations)
{
    long number_in_circle=0;
    long toss;
    unsigned int seed;
    double x,y,distance_squared;
    double pi_estimate;

    seed=time(NULL);

    for(toss=0;toss<iterations;toss++){
        x=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);
        y=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);

        distance_squared=x*x+y*y;
        if(distance_squared<=1) number_in_circle++;
    }
    pi_estimate=4*number_in_circle/((double)iterations);

    return pi_estimate;
}


// Find how close the calculation is to the actual value of PI
double getDifference(double calculatedPi)
{
  return calculatedPi - 3.14159265358979323846;
}


// TODO: You need to implement an OpenMP parallel version using atomics
//       Use rand_r() for thread safe random number generation.
//       More details about rand_r() is here: http://linux.die.net/man/3/rand_r
//       You must also make sure that each thread start with a DIFFERENT SEED!
double parallelCompute_atomic(long iterations, int numberOfThreads)
{
    long number_in_circle=0;
    long i;

    double x,y;
    double distance_squared;
    double pi_estimate;

    unsigned int seed;

#pragma omp parallel num_threads(numberOfThreads)\
    private(x,y,distance_squared,seed)
    {
        long my_number=0.0;
        seed=omp_get_thread_num();

#pragma omp for
        for(i=0;i<iterations;i++){
            x=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);
            y=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);

            distance_squared=x*x+y*y;
            if(distance_squared<=1) my_number++;
        }
#pragma omp atomic
        number_in_circle+=my_number;

    }

    pi_estimate=4*number_in_circle/((double)iterations);

    return pi_estimate;
}


// TODO: Same as the other OpenMP version above,
// but uses OpenMP reduction clause to aggregate partial results
double parallelCompute_reduction(long iterations, int numberOfThreads)
{
    long number_in_circle=0;
    long i;

    double x,y;
    double distance_squared;
    double pi_estimate;

    unsigned int seed;

#pragma omp parallel num_threads(numberOfThreads)\
    private(x,y,distance_squared,seed) reduction(+:number_in_circle)
    {
        long my_number=0.0;
        seed=omp_get_thread_num();

#pragma omp for
        for(i=0;i<iterations;i++){
            x=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);
            y=((double)rand_r(&seed)-(double)RAND_MAX/2)/(double)(RAND_MAX/2);

            distance_squared=x*x+y*y;
            if(distance_squared<=1) my_number++;
        }

        number_in_circle+=my_number;

    }

    pi_estimate=4*number_in_circle/((double)iterations);

    return pi_estimate;
}

