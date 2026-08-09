/*
 * nn.cu
 * Nearest Neighbor
 * Modified by André Du Bois: changed depracated api, creating data set in memory. clean up code not used
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#include <time.h>


__device__ static double atomic_cas(double* address, double oldv, double newv)
{
    unsigned long long int * address_as_i = (unsigned long long int *) address;
    return  __longlong_as_double(  atomicCAS(address_as_i, __double_as_longlong(oldv),__double_as_longlong(newv))  );
}



__device__
double euclid(double *d_locations, float lat, float lng)
{
return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}


extern "C" __global__ void map_step_2para_1resp_kernel(double *d_array, double *d_result, int step, float par1, float par2, int size)
{
	int globalId = (threadIdx.x + (blockIdx.x * blockDim.x));
	int id = (step * globalId);
if((globalId < size))
{
	d_result[globalId] = euclid((d_array + id), par1, par2);
}

}




__device__
double menor(double x, double y)
{
if((x < y))
{
return (x);
}
else{
return (y);
}

}


extern "C" __global__ void reduce_kernel(double *a, double *ref4, int n)
{
__shared__ double cache[256];
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	int cacheIndex = threadIdx.x;
	double temp = ref4[0];
while((tid < n)){
	temp = menor(a[tid], temp);
	tid = ((blockDim.x * gridDim.x) + tid);
}
	cache[cacheIndex] = temp;
__syncthreads();
	int i = (blockDim.x / 2);
while((i != 0)){
if((cacheIndex < i))
{
	cache[cacheIndex] = menor(cache[(cacheIndex + i)], cache[cacheIndex]);
}

__syncthreads();
	i = (i / 2);
}
if((cacheIndex == 0))
{
	double current_value = ref4[0];
while((! (current_value == atomic_cas(ref4, current_value, menor(cache[0], current_value))))){
	current_value = ref4[0];
}
}

}




void loadData(double *locations, int size);
//void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
//void printUsage();
//int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
//                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position

__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}
**/
/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
//	int    i=0;
	//float lat=0, lng=0;
	
  //  std::vector<Record> records;
	double *locations;

  int numRecords = atoi(argv[1]);
    
   locations = (double *)malloc(sizeof(double) * 2*numRecords);
   // int numRecords = loadData(filename,records,locations);
   loadData(locations,numRecords);

    
	double *distances;
	//Pointers to device memory
	double *d_locations;
	double *d_distances;


	

	/**
	
	* Allocate memory on host and device

  */

  float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    //int size_dist = numRecords/2;
	distances = (double *)malloc(sizeof(double) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(double) * 2 * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(double) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(double) * 2* numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel --
    */

   

    map_step_2para_1resp_kernel<<< numRecords, 1 >>>(d_locations,d_distances,2,0.0,0.0, numRecords);
    

    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRecords + threadsPerBlock - 1)/ threadsPerBlock;

    double *resp, *d_resp;
    resp = (double *)malloc(sizeof(double));
    resp[0] = 50000;
	cudaMalloc((void **) &d_resp,sizeof(double));
    cudaMemcpy( d_resp, resp, sizeof(double) , cudaMemcpyHostToDevice);

    reduce_kernel<<< blocksPerGrid, threadsPerBlock >>>(d_distances,d_resp,numRecords);
    cudaDeviceSynchronize();
    //Copy data from device memory to host memory

    cudaMemcpy( resp, d_resp, sizeof(double), cudaMemcpyDeviceToHost );


	// find the resultsCount least distances
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);
   
     cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

     printf("CUDA\t%d\t%3.1f\n", numRecords,time);

}

void loadData(double* locations, int size){
   
	for (int i=0;i<size;i++){
			
            locations[0] = ((double)(7 + rand() % 63)) + ((double) rand() / (double) 0x7fffffff);

            locations[1] = ((double)(rand() % 358)) + ((double) rand() / (double) 0x7fffffff); 

            locations = locations +2;
            
           
        }
     
}



