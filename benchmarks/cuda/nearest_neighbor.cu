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

__device__ static float atomic_cas(float* address, float oldv, float newv)
{
    int* address_as_i = (int*) address;
    return  __int_as_float(atomicCAS(address_as_i, __float_as_int(oldv), __float_as_int(newv)));
}



__device__
float euclid(float *d_locations, float lat, float lng)
{
return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}


extern "C" __global__ void map_step_2para_1resp_kernel(float *d_array, float *d_result, int step, float par1, float par2, int size)
{
	int globalId = (threadIdx.x + (blockIdx.x * blockDim.x));
	int id = (step * globalId);
if((globalId < size))
{
	d_result[globalId] = euclid((d_array + id), par1, par2);
}

}




__device__
float menor(float x, float y)
{
if((x < y))
{
return (x);
}
else{
return (y);
}

}


extern "C" __global__ void reduce_kernel(float *a, float *ref4, int n)
{
__shared__ float cache[256];
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	int cacheIndex = threadIdx.x;
	float temp = ref4[0];
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
	float current_value = ref4[0];
while((! (current_value == atomic_cas(ref4, current_value, menor(cache[0], current_value))))){
	current_value = ref4[0];
}
}

}


void loadData(float *locations, int size);

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
//	int    i=0;
	//float lat=0, lng=0;
	
  //  std::vector<Record> records;
	float *locations;

  int numRecords = atoi(argv[1]);
    
   locations = (float *)malloc(sizeof(float) * 2*numRecords);
   // int numRecords = loadData(filename,records,locations);
   loadData(locations,numRecords);

    
	float *distances;
	//Pointers to device memory
	float *d_locations;
	float *d_distances;


	

	/**
	
	* Allocate memory on host and device

  */

  float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    //int size_dist = numRecords/2;
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(float) * 2 * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(float) * 2* numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel --
    */

   

    map_step_2para_1resp_kernel<<< numRecords, 1 >>>(d_locations,d_distances,2,0.0,0.0, numRecords);
    

    cudaDeviceSynchronize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRecords + threadsPerBlock - 1)/ threadsPerBlock;

    float *resp, *d_resp;
    resp = (float *)malloc(sizeof(float));
    resp[0] = 50000;
	cudaMalloc((void **) &d_resp,sizeof(float));
    cudaMemcpy( d_resp, resp, sizeof(float) , cudaMemcpyHostToDevice);

    reduce_kernel<<< blocksPerGrid, threadsPerBlock >>>(d_distances,d_resp,numRecords);
    cudaDeviceSynchronize();
    //Copy data from device memory to host memory

    cudaMemcpy( resp, d_resp, sizeof(float), cudaMemcpyDeviceToHost );


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

void loadData(float* locations, int size){
   
	for (int i=0;i<size;i++){
			
            locations[0] = ((float)(7 + rand() % 63)) + ((float) rand() / (float) 0x7fffffff);

            locations[1] = ((float)(rand() % 358)) + ((float) rand() / (float) 0x7fffffff); 

            locations = locations +2;
            
           
        }
     
}



