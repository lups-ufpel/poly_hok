#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

ErlNifResourceType *KERNEL_TYPE;
ErlNifResourceType *ARRAY_TYPE;

void
dev_array_destructor(ErlNifEnv *env, void *res) {
  CUdeviceptr *dev_array = (CUdeviceptr*) res;
  cuMemFree(*dev_array);
}


static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  KERNEL_TYPE =
  enif_open_resource_type(env, NULL, "kernel", NULL, ERL_NIF_RT_CREATE  , NULL);
  ARRAY_TYPE =
  enif_open_resource_type(env, NULL, "gpu_ref", dev_array_destructor, ERL_NIF_RT_CREATE  , NULL);
  return 0;
}

static ERL_NIF_TERM new_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int data_size;
  int nrow,ncol;
  ERL_NIF_TERM term;
  
  CUresult err;
  CUdeviceptr dev_array;
  
  if (!enif_get_int(env, argv[0], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }
  char type_name[1024];
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0) 
  {

    
    data_size = nrow * ncol * sizeof(float);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL


   
  } else if (strcmp(type_name, "int") == 0)
  {
    data_size = nrow * ncol * sizeof(int);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL


    
  } else if (strcmp(type_name, "double") == 0)
  {
   data_size = nrow * ncol * sizeof(double);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif:");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL


     
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error new_gpu_array_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }

     CUdeviceptr *gpu_res = (CUdeviceptr*)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
     *gpu_res = dev_array;
      term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
      enif_release_resource(gpu_res);
  return term;
}

static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  char type_name[1024];

  ERL_NIF_TERM  result;
  CUdeviceptr dev_array;
  CUresult err;


  CUdeviceptr *array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
       return enif_make_badarg(env);
    }

  dev_array = *array_res;

   
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

  
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  
  if (strcmp(type_name, "float") == 0) 
  {
    

    int result_size = sizeof(float) * (nrow*ncol);
    int data_size = sizeof(float) * (nrow*ncol);
    float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

    float *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    err = cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error get_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

  } else if (strcmp(type_name, "int") == 0)
  {
    
    int result_size = sizeof(int) * (nrow*ncol);
    int data_size = sizeof(int) * (nrow*ncol);
    int *result_data = (int *) enif_make_new_binary(env, result_size, &result);

    int *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    err=  cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error get_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

    
  } else if (strcmp(type_name, "double") == 0)
  {
   
    int result_size = sizeof(double) * (nrow*ncol);
    int data_size = sizeof(double) * (nrow*ncol);
    double *result_data = (double *) enif_make_new_binary(env, result_size, &result);

    double *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    err=  cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error get_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

    
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error get_nx_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }
  return result;
}


static ERL_NIF_TERM create_gpu_array_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  array_el;
  
  int nrow,ncol;

  CUresult err;
  CUdeviceptr     dev_array;
  
  if (!enif_inspect_binary(env, argv[0], &array_el)) return enif_make_badarg(env);

  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

 char type_name[1024];
  
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  // FINAL TERM TO BE RETURNED:
  ERL_NIF_TERM term;

  if (strcmp(type_name, "float") == 0) 
  { 
    
    float         *array;
    
    array = (float *) array_el.data;

   int data_size = sizeof(float)* ncol*nrow;

    ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
  
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL

    

  } 
  else if (strcmp(type_name, "int") == 0)
  {
    int         *array;
    array = (int *) array_el.data;
    
    int data_size = sizeof(int)* ncol*nrow;

     ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
  
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL



  }
  else if (strcmp(type_name, "double") == 0)
  {
    double        *array;
 
    array = (double *) array_el.data;

    int data_size = sizeof(double)* ncol*nrow;
    
    err = cuMemAlloc(&dev_array, data_size) ;
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL

     
  }
 /* more else if clauses */
 else /* default: */
 {

  
    char message[200];
        strcpy(message,"Error create_gpu_array_nx_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }

  CUdeviceptr *gpu_res = (CUdeviceptr*)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
  *gpu_res = dev_array;
  term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);
  return term;
}




static ErlNifFunc nif_funcs[] = {
    {"new_gpu_array_nif", 3, new_gpu_array_nif},
    {"get_gpu_array_nif", 4, get_gpu_array_nif},
    {"create_gpu_array_nx_nif", 4, create_gpu_array_nx_nif}
   
};

ERL_NIF_INIT(Elixir.Hok, nif_funcs, &load, NULL, NULL, NULL)
