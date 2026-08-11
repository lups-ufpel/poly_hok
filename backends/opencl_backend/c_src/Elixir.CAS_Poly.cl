inline int cas_int(volatile __global int *address, int oldv, int newv) {
    return atomic_cmpxchg(address, oldv, newv);
}

/*
    For the float and double compare-and-swap (CAS) operations, we are using a
   trick: we are reinterpreting the float/double values as integer/long integer
   values.

    Since the OpenCL standard only provides atomic operations for 32 and 64 bit
   integers, this is necessary and works perfectly fine.

    Since we only care if the value was changed or not, we can safely use this
   method, as it doesn't require mathematical operations on the float/double
   values themselves.
*/

inline float cas_float(volatile __global float *address, float oldv, float newv) {
    volatile __global uint *i_address = (volatile __global uint *)address;
    uint i_oldv = as_uint(oldv);
    uint i_newv = as_uint(newv);

    uint i_res = atomic_cmpxchg(i_address, i_oldv, i_newv);
    
    // Return the float representation of the result
    return as_float(i_res);
}

#if DOUBLE_SUPPORTED

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline double cas_double(volatile __global double *address, double oldv, double newv) {
    volatile __global ulong *l_address = (volatile __global ulong *)address;
    ulong l_oldv = as_ulong(oldv);
    ulong l_newv = as_ulong(newv);

    // For 64-bit atomic compare-and-swap, the function is atom_cmpxchg (weird, but this is the specification)
    ulong l_res = atom_cmpxchg(l_address, l_oldv, l_newv);
    
    // Return the double representation of the result
    return as_double(l_res);
}

#endif