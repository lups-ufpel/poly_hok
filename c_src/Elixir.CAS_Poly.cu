

__device__ static float cas_float(float* address, float oldv, float newv)
{
    int* address_as_i = (int*) address;
    return  __int_as_float(atomicCAS(address_as_i, __float_as_int(oldv), __float_as_int(newv)));
}

__device__ static int cas_int(int* address, int oldv, int newv)
{
    return  atomicCAS(address, oldv, newv);
}

__device__ static double cas_double(double* address, double oldv, double newv)
{
    unsigned long long int * address_as_i = (unsigned long long int *) address;
    return  __longlong_as_double(  atomicCAS(address_as_i, __double_as_longlong(oldv),__double_as_longlong(newv))  );
}

