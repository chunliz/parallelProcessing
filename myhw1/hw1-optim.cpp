#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define MIN(x,y) (x<y)?x:y

#define N 10000
#define M 10000
#define Nblock_size 8
#define Mblock_size 3860

int main(){
	double x[N];
	double y[M];
	double a[N][M];
	
	int i,j;
	int k,k_start,k_end;
	int l,l_start,l_end;
	int Nb,Mb;

	if(N%Nblock_size==0) Nb=N/Nblock_size;
	else Nb=N/Nblock_size+1;

	if(M%Mblock_size==0) Mb=M/Mblock_size;
	else Mb=M/Mblock_size+1;

	srand((unsigned)time(0));

	for(i=0;i<N;i++){
	x[i]=(rand()%100)+1;
	}
	for(j=0;j<M;j++){
	y[j]=(rand()%100)+1;
	}


	double time1=omp_get_wtime();

	for(i=1;i<=Nb;i++){
		for(j=1;j<=Mb;j++){
			k_start=(i-1)*Nblock_size;k_end=MIN(i*Nblock_size-1,N);
			l_start=(i-1)*Mblock_size;l_end=MIN(i*Mblock_size-1,M);
			for(k=k_start;k<=k_end;k++){
				for(l=l_start;l<=l_end;l++){
				a[k][l]=x[k]*y[l];
				}
			}
		}
	}

	double time2=omp_get_wtime();

	cout << "total time = " << time2-time1 << endl;
	return 0;
}

