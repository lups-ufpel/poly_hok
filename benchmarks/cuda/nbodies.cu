#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__
void gpu_nBodies(double *p, double *c, int n)
{
	float softening = 1.0e-9;
	float dt = 0.01;
	float fx = 0.0;
	float fy = 0.0;
	float fz = 0.0;
for( int j = 0; j<n; j++){
	double dx = (c[(6 * j)] - p[0]);
	double dy = (c[((6 * j) + 1)] - p[1]);
	double dz = (c[((6 * j) + 2)] - p[2]);
	double distSqr = ((((dx * dx) + (dy * dy)) + (dz * dz)) + softening);
	float invDist = (1.0 / sqrt(distSqr));
	float invDist3 = ((invDist * invDist) * invDist);
	fx = (fx + (dx * invDist3));
	fy = (fy + (dy * invDist3));
	fz = (fz + (dz * invDist3));
}

	p[3] = (p[3] + (dt * fx));
	p[4] = (p[4] + (dt * fy));
	p[5] = (p[5] + (dt * fz));
}



__device__
void gpu_integrate(double *p, float dt, int n)
{
	p[0] = (p[0] + (p[3] * dt));
	p[1] = (p[1] + (p[4] * dt));
	p[2] = (p[2] + (p[5] * dt));
}


__global__ void map1(double *d_array, int step, double *par1, int par2, int size)
{
	int globalId = ((blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x)) + threadIdx.x);
	int id = (step * globalId);
if((globalId < size))
{
gpu_nBodies((d_array + id), par1, par2);
}

}

__global__ void map2(double *d_array, int step, float par1, int par2, int size)
{
	int globalId = ((blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x)) + threadIdx.x);
	int id = (step * globalId);
if((globalId < size))
{
gpu_integrate((d_array + id), par1, par2);
}

}



void randomizeBodies(double *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = (double) (rand() / (float)RAND_MAX);
  }
}



int main(const int argc, const char** argv) {
  
  int user_value = atoi(argv[1]);

  int nBodies = user_value;
  int block_size =  128;
 // float softening = 0.000000001;
  cudaError_t nb_error;
  
  //const float dt = 0.01; // time step
  

  int bytes = nBodies*sizeof(double)*6;
  double *h_buf = (double*)malloc(bytes);
  double *d_resp = (double*)malloc(bytes);
 

  randomizeBodies(h_buf, 6*nBodies); // Init pos / vel data

  double *d_buf;

  
  

  int nBlocks = (nBodies + block_size - 1) / block_size;
  
    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;
  //////////////////////////////// 

///////////////////
  cudaMalloc(&d_buf, bytes);
  nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
  //////// 

  cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);
   nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
  //////// 
 

   ////////////////////
    map1<<<nBlocks, block_size>>>(d_buf,6,d_buf, nBodies,nBodies);
  
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
  //////// 
 
    cudaDeviceSynchronize();
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
  ////////


   map2<<<nBlocks, block_size>>>(d_buf, 6, 0.01, nBodies,nBodies); // compute interbody forces
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
   

   /////////////////////////////
   cudaMemcpy(d_resp, d_buf, bytes, cudaMemcpyDeviceToHost);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
  //////// 
 
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("cuda\t%d\t%3.1f\n", nBodies,time);

    /*
    begin = clock();
    cpu_bodyForce(h_buf,dt,nBodies,softening);
    float *p = h_buf;
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[6*i] += p[6*i+3]*dt;
      p[6*i+1] += p[6*i+4]*dt;
      p[6*i+2] += p[6*i+5]*dt;
    }
  
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("CPU elapsed time is %f seconds\n", time_spent);


    for (int i = 0 ; i < nBodies; i++) { // integrate position
      if (!Equality(h_buf[i],d_resp[i]))
        { printf("Diferentes h_buf[%d] = %f, d_resp[%i] = %f \n",i,h_buf[i],i,d_resp[i]); }
    }
*/
    free(h_buf);
    free(d_resp);
    cudaFree(d_buf);
}
