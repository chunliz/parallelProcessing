#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define MIN(x,y) (x<y)?x:y

#define N 10000
#define M 10000

int main(){
	double x[N];
	double y[M];
	double a[N][M];
	
	int i,j;

	srand((unsigned)time(0));

	for(i=0;i<N;i++){
	x[i]=(rand()%100)+1;
}
	for(j=0;j<M;j++){
	y[j]=(rand()%100)+1;
}

	double time1=omp_get_wtime();

	for(i=0;i<N;i++){
		for(j=0;j<M;j++){
			a[i][j]=x[i]*y[j];
		}
}

	double time2=omp_get_wtime();

	cout << "total time = " << time2-time1 << endl;
	return 0;
}

