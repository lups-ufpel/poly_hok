

__device__ static int atomic_cas(int* address, int oldv, int newv)
{
    return  atomicCAS(address, oldv, newv);
}


