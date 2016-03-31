#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <cmath>

using namespace std;

vector<double> S; // this is the input vector
double global_min = 1000000000000.0;
double main_sum;

double VectorSum(vector<double> &V){
	double sum=0;
	for(int i=0; i < V.size(); i++){
		sum += V[i];
	}
	return sum;
}

//index is the index of the new element to be added into S
void GenerateSubset(int index, vector<double> &S, vector<double> &subset) {

    if (index == S.size()-1) { //then this subset is done
		double diff1 = abs(VectorSum(subset) - main_sum/2);
		if(diff1 < global_min) global_min = diff1;
		double diff2 = abs(VectorSum(subset) + S[index] - main_sum/2);
		if(diff2 < global_min){
			global_min = diff2;
			subset.push_back(S[index]);
		}
		return;
	}
	
    
    // this is only a pseudocode!
    // you will need to make the appropriate modifications
    subset.push_back(S[index]); 
	//add the index element in subset
    GenerateSubset(index+1, S, subset);  //generate subsets with the index element in them
    subset.pop_back();
    GenerateSubset(index+1, S, subset);  //generate subsets without the index element
	
	return;
}



int main(int argc, char* argv[]) {
  double start_time, end_time, time_diff;
  int i, nweights, nthreads;
  vector<double> subset; // this is where you will generate the subsets
  subset.clear();
 
  // Get the number of weights and number of threads from the command line
  if (argc == 3) {
    nweights = strtol(argv[1], NULL, 10);
    nthreads = strtol(argv[2], NULL, 10);
  }
  else {
    printf("\nWhen running this program, please include number of weights and number of threads on command line.\n");
    return 0;
  }
  
  printf("\nnumber of weigts: %d\n", nweights);
  printf("number of threads: %d\n", nthreads);
    
  main_sum = 0;
  //srand(time(NULL));
  srand(0);
  for (i = 0 ; i < nweights; i++) {
    S.push_back(((double)rand())/RAND_MAX);
    main_sum += S[i];
  }
  
  printf("main set : ");
  for (i = 0 ; i < S.size() ; i++)
    printf("%lf ", S[i]);
  printf("\n");
  printf("main sum = %lf\n", main_sum);
  
  start_time = omp_get_wtime();

  GenerateSubset(0, S, subset);
  
  end_time = omp_get_wtime();
  
  printf("\n minimum diff = %.14lf\n", 2*global_min);
  
  printf("time needed = %f\n", end_time - start_time);
}
