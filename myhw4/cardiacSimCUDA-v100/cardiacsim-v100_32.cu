/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuda.h"

#define BLOCK_WIDTH 32

using namespace std;


// Utilities
// 

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            cerr << "ERROR: Bad call to gettimeofday" << endl;
            return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

/*
// Allocate a 2D array
double **alloc2D(int m,int n){
   double **E;
   int nx=n, ny=m;
   E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
   assert(E);
   int j;
   for(j=0;j<ny;j++) 
     E[j] = (double*)(E+ny) + j*nx;
   return(E);
}
*/
    
// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
 double stats(double *E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
       l2norm += E[j*(n+2)+i]*E[j*(n+2)+i];
       if (E[j*(n+2)+i] > mx)
           mx = E[j*(n+2)+i];
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }

// External functions
extern "C" {
    void splot(double *E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], float& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);

__global__ void boundaryKernal(double *E_prev, int n, int m){
  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((col == 0) && (row < m))
        E_prev[(row+1)*(n+2)] = E_prev[(row+1)*(n+2)+2];
    if ((col == (n-1)) && (row < m))
        E_prev[(row+1)*(n+2)+n+1] = E_prev[(row+1)*(n+2)+n-1];
    if ((row == 0) && (col < n))
        E_prev[col+1] = E_prev[col+1+2*(n+2)];
    if ((row == (m-1)) && (col < n))
        E_prev[(m+1)*(n+2)+col+1] = E_prev[(m-1)*(n+2)+col+1];
    __syncthreads();
}


__global__ void PDEKernal(double *E, double *E_prev, int n, int m, double alpha){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if((row < m) && (col < n)){
    
        E[(row+1)*(n+2)+col+1] = E_prev[(row+1)*(n+2)+col+1]+alpha*(E_prev[(row+1)*(n+2)+col+2]+E_prev[(row+1)*(n+2)+col]-4*E_prev[(row+1)*(n+2)+col+1]+E_prev[(row+2)*(n+2)+col+1]+E_prev[row*(n+2)+col+1]);
    }
    __syncthreads();

}

__global__ void ODEKernal1(double *E, double *R, int n, int m, double kk, double dt, double a){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if((row < m) && (col < n)){
    
        E[(row+1)*(n+2)+col+1] = E[(row+1)*(n+2)+col+1] -dt*(kk* E[(row+1)*(n+2)+col+1]*(E[(row+1)*(n+2)+col+1] - a)*(E[(row+1)*(n+2)+col+1]-1)+ E[(row+1)*(n+2)+col+1] *R[(row+1)*(n+2)+col+1]);
    }
    __syncthreads();
}

__global__ void ODEKernal2(double *E, double *R, int n, int m, double kk, double dt, double b,  double epsilon, double M1, double M2){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if((row < m) && (col < n)){
        R[(row+1)*(n+2)+col+1] = R[(row+1)*(n+2)+col+1] + dt*(epsilon+M1* R[(row+1)*(n+2)+col+1]/( E[(row+1)*(n+2)+col+1]+M2))*(-R[(row+1)*(n+2)+col+1]-kk* E[(row+1)*(n+2)+col+1]*(E[(row+1)*(n+2)+col+1]-b-1));
        
    }
    __syncthreads();
}

__global__ void swapKernal(double *E, double *E_prev, int n, int m){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp = 0.0;
    if ((row < m) && (col < n)){
        tmp = E[(row+1)*(n+2)+col+1];
        E[(row+1)*(n+2)+col+1] = E_prev[(row+1)*(n+2)+col+1];
        E_prev[(row+1)*(n+2)+col+1] = tmp;
    }
    __syncthreads();
    
}


/*
void simulate (double* E,  double* E_prev,double* R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b)
{
  int i, j; 
    /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
/*

    for (j=1; j<=m; j++) 
      E_prev[j*(m+2)] = E_prev[j*(m+2)+2];
    for (j=1; j<=m; j++) 
      E_prev[j*(m+2)+n+1] = E_prev[j*(m+2)+n-1];
    
    for (i=1; i<=n; i++) 
      E_prev[i] = E_prev[2*(m+2)+i];
    for (i=1; i<=n; i++) 
      E_prev[(m+1)*(m+2)+i] = E_prev[(m-1)*(m+2)+i];

    
    // Solve for the excitation, the PDE
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++) {
    E[j*(m+2)+i] = E_prev[j*(m+2)+i]+alpha*(E_prev[j*(m+2)+i+1]+E_prev[j*(m+2)+i-1]-4*E_prev[j*(m+2)+i]+E_prev[(j+1)*(m+2)+i]+E_prev[(j-1)*(m+2)+i]);
      }
    }
    
    /* 
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */

/*
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
    E[j*(m+2)+i] = E[j*(m+2)+i] -dt*(kk* E[j*(m+2)+i]*(E[j*(m+2)+i] - a)*(E[j*(m+2)+i]-1)+ E[j*(m+2)+i] *R[j*(m+2)+i]);
    }
    
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
    R[j*(m+2)+i] = R[j*(m+2)+i] + dt*(epsilon+M1* R[j*(m+2)+i]/( E[j*(m+2)+i]+M2))*(-R[j*(m+2)+i]-kk* E[j*(m+2)+i]*(E[j*(m+2)+i]-b-1));
    }
    
}

*/

// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double *E, *R, *E_prev;
  
  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
  
  float T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;
  int num_threads=1; 

  cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads);
  m = n;  
  int size=(m+2)*(n+2);

  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the 
  // boundaries of the computation box

  E=(double *)malloc(size*sizeof(double));
  E_prev=(double *)malloc(size*sizeof(double));
  R=(double *)malloc(size*sizeof(double));
  
  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j*(n+2)+i] = R[j*(n+2)+i] = 0;
  
  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j*(n+2)+i] = 1.0;
  
  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j*(n+2)+i] = 1.0;
  
  double dx = 1.0/n;

  // For time integration, these values shouldn't change 
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  cout << "Grid Size       : " << n << endl; 
  cout << "Duration of Sim : " << T << endl; 
  cout << "Time step dt    : " << dt << endl; 
  cout << "Process geometry: " << px << " x " << py << endl;
  if (no_comm)
    cout << "Communication   : DISABLED" << endl;
  
  cout << endl;
  
  // Start the timer
  double t0 = getTime();
  
 
  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;

  double *d_E, *d_E_prev, *d_R;

  cudaMalloc((void **) &d_E, size*sizeof(double));
  cudaMemcpy(d_E, E, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_E_prev, size*sizeof(double));
  cudaMemcpy(d_E_prev, E_prev, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_R, size*sizeof(double));
  cudaMemcpy(d_R, R, size*sizeof(double), cudaMemcpyHostToDevice);

  int tnx=n/BLOCK_WIDTH;
  int tny=m/BLOCK_WIDTH;

  if(n%BLOCK_WIDTH) tnx++;
  if(m%BLOCK_WIDTH) tny++;

  dim3 dimGrid(tnx,tny,1);
  dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);

  double t1 = getTime();
  
  while (t<T) {
    
    t += dt;
    niter++;

    boundaryKernal<<<dimGrid,dimBlock>>>(d_E_prev, n, m);
    cudaDeviceSynchronize();

    PDEKernal<<<dimGrid,dimBlock>>>(d_E, d_E_prev, n, m, alpha);
    cudaDeviceSynchronize();

    ODEKernal1<<<dimGrid,dimBlock>>>(d_E, d_R, n, m, kk, dt, a);
    cudaDeviceSynchronize();

    ODEKernal2<<<dimGrid,dimBlock>>>(d_E, d_R, n, m, kk, dt, b, epsilon, M1, M2);
    cudaDeviceSynchronize();
 
    //simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

    swapKernal<<<dimGrid,dimBlock>>>(d_E, d_E_prev, n, m);
    cudaDeviceSynchronize();
    
    //swap current E with previous E
    //double *tmp = E; E = E_prev; E_prev = tmp;
    
    if (plot_freq){

      int k = (int)(t/plot_freq);
      if ((t - k * plot_freq) < dt){
        cudaMemcpy(E, d_E, size*sizeof(double), cudaMemcpyDeviceToHost);
	splot(E,t,niter,m+2,n+2);
      }
    }
  }//end of while loop

  double time_elapsed1 = getTime() - t1;

  cudaMemcpy(E, d_E, size*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(E_prev, d_E_prev, size*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(R, d_R, size*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_E);
  cudaFree(d_E_prev);
  cudaFree(d_R);

  double time_elapsed = getTime() - t0;

  double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
  double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;
  cout << "BLOCK SIZE                  : " << BLOCK_WIDTH << endl;
  cout << "Number of Iterations        : " << niter << endl;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
  cout << "Time (no data transfer)     : " << time_elapsed1 <<endl;
  cout << "Sustained Gflops Rate       : " << Gflops << endl; 
  cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;
  cout << "************************************************************" <<endl;
  cout << "************************************************************" <<endl;
  cout << "************************************************************" <<endl;

  //for(i=1;i<=m;i++) printf("E[%d,%d]:%f\n",i,i,E_prev[i*(m+2)+i]);

  double mx;
  double l2norm = stats(E_prev,m,n,&mx);
  cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

  if (plot_freq){
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }
  
  free (E);
  free (E_prev);
  free (R);
  
  return 0;
}
