#include <erl_nif.h>

#include <string>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

ErlNifResourceType *ARRAY_TYPE;
ErlNifResourceType *COMPILED_PTX;

typedef struct _compiled_kernel
{
  char *ptx;
  std::string kernel_name;
} CompiledKernel;

void dev_array_destructor(ErlNifEnv * /* env */, void *res)
{
  CUdeviceptr *dev_array = (CUdeviceptr *)res;
  cuMemFree(*dev_array);
}

void compiled_ptx_destructor(ErlNifEnv * /* env */, void *res)
{
  CompiledKernel *compiled_kernel = (CompiledKernel *)res;
  delete[] compiled_kernel->ptx;
}

CUcontext context = NULL;
void init_cuda(ErlNifEnv *env)
{
  if (context == NULL)
  {
    CUresult err;
    int device = 0;
    cuInit(0);

    err = cuCtxCreate(&context, NULL, 0, device);

    if (err != CUDA_SUCCESS)
    {
      const char *error;
      cuGetErrorString(err, &error);

      std::string message = "Error initializing CUDA context: " + std::string(error);

      enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
    }
  }
  else
  {
    cuCtxSetCurrent(context);
  }
}

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info)
{
  ARRAY_TYPE = enif_open_resource_type(
      env,
      NULL,
      "gpu_ref",
      dev_array_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  COMPILED_PTX = enif_open_resource_type(
      env,
      NULL,
      "compiled_ptx",
      compiled_ptx_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Initialize CUDA context
  init_cuda(env);

  return 0;
}

// This function is called when the NIF library is unloaded
static void unload(ErlNifEnv * /* env */, void * /* priv_data */)
{
  if (context != NULL)
  {
    cuCtxDestroy(context);
  }
}

// This function retrieves the CUDA GPU array from VRAM to the host as an Erlang term.
static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 4)
  {
    return enif_make_badarg(env);
  }

  int nrow, ncol;
  char type_name[1024];

  CUdeviceptr dev_array, *p_dev_array;
  CUresult err;

  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&p_dev_array))
  {
    return enif_make_badarg(env);
  }
  dev_array = *p_dev_array;

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  size_t data_size;
  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * (nrow * ncol);
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * (nrow * ncol);
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * (nrow * ncol);
  }
  else
  {
    std::string message = "Error get_gpu_array_nif: unknown type: " + std::string(type_name);
    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  // Allocate memory in the host to store the result
  // According to Erlang's docs, for LARGE binaries, it is better to use enif_alloc_binary.
  ErlNifBinary host_bin;

  if (!enif_alloc_binary(data_size, &host_bin))
  {
    std::string message =
        "[ERROR] (get_gpu_array_nif) failed to allocate binary of size " +
        std::to_string(data_size);
    return enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  //// MAKE CUDA CALL
  err = cuMemcpyDtoH((void *)host_bin.data, dev_array, data_size);

  if (err != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(err, &error);

    std::string message =
        "Error (get_gpu_array_nif): error copying data from device to host: " +
        std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }
  //////// END CUDA CALL

  ERL_NIF_TERM result = enif_make_binary(env, &host_bin);
  return result;
}

// This function creates a new GPU array with the specified number of rows, columns, and type.
// It allocates memory on the GPU and copies data to it from the host array provided.
static ERL_NIF_TERM new_gpu_array_from_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 4)
  {
    return enif_make_badarg(env);
  }

  int nrow, ncol;
  ErlNifBinary host_array_el;

  // Get the host array binary
  if (!enif_inspect_binary(env, argv[0], &host_array_el))
  {
    return enif_make_badarg(env);
  }

  // Get rows and columns
  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }
  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  size_t data_size;

  // Calculates the size of the data to be copied to the GPU
  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * ncol * nrow;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * ncol * nrow;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * ncol * nrow;
  }
  else
  {
    std::string message = "Error (new_gpu_array_from_nx_nif): unknown type: " + std::string(type_name);
    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  CUresult err;
  CUdeviceptr dev_array;

  // Allocating memory on the GPU for the new array
  err = cuMemAlloc(&dev_array, data_size);
  if (err != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(err, &error);

    std::string message = "Error (new_gpu_array_from_nx_nif:) cuMemAlloc size: " + std::to_string(data_size) + " : " + std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  // Copying data from the host array to the newly allocated GPU array
  err = cuMemcpyHtoD(dev_array, (void *)host_array_el.data, data_size);
  if (err != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(err, &error);

    std::string message = "Error (new_gpu_array_from_nx_nif:) cuMemcpyHtoD size: " + std::to_string(data_size) + " : " + std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  // Allocating resource for the GPU array to be returned to Erlang
  CUdeviceptr *gpu_res = (CUdeviceptr *)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));

  // Storing the device pointer in the resource
  *gpu_res = dev_array;

  // Creating an Erlang term from the resource and releasing the resource so that it will be freed when Erlang garbage collects it
  ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);
  enif_release_resource(gpu_res);

  return return_term;
}

// Creates a new empty GPU array with the specified number of rows, columns, and type
static ERL_NIF_TERM new_empty_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 3)
  {
    return enif_make_badarg(env);
  }

  int nrow, ncol;
  ErlNifBinary host_array_el;

  // Get rows and columns
  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }
  if (!enif_get_int(env, argv[1], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  size_t data_size;

  // Calculates the size of the data to be allocated on the GPU based on the type and dimensions
  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * ncol * nrow;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * ncol * nrow;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * ncol * nrow;
  }
  else
  {
    std::string message = "Error (new_empty_gpu_array_nif): unknown type: " + std::string(type_name);
    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  CUresult err;
  CUdeviceptr dev_array;

  // Allocating memory on the GPU for the new array
  err = cuMemAlloc(&dev_array, data_size);
  if (err != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(err, &error);

    std::string message = "Error (new_empty_gpu_array_nif:) cuMemAlloc size: " + std::to_string(data_size) + " : " + std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  // Allocating resource for the GPU array to be returned to Erlang
  CUdeviceptr *gpu_res = (CUdeviceptr *)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));

  // Storing the device pointer in the resource
  *gpu_res = dev_array;

  // Creating an Erlang term from the resource and releasing the resource so that it will be freed when Erlang garbage collects it
  ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);
  enif_release_resource(gpu_res);

  return return_term;
}

static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  CUresult r = cuCtxSynchronize();

  if (r != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(r, &error);

    std::string message = "Error (synchronize_nif): error synchronizing device: " + std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  return enif_make_int(env, 0);
}

ERL_NIF_TERM fail_cuda(ErlNifEnv *env, CUresult result, const char *obs)
{
  const char *error;
  cuGetErrorString(result, &error);

  std::string message =
      "Error CUDA " +
      std::string(obs) +
      ": " +
      std::string(error);

  return enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
}

ERL_NIF_TERM fail_nvrtc(ErlNifEnv *env, nvrtcResult result, const char *obs)
{
  std::string message =
      "Error  NVRTC " +
      std::string(obs) +
      ": " +
      std::string(nvrtcGetErrorString(result));

  return enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
}

ERL_NIF_TERM compile_to_ptx(ErlNifEnv *env, std::string &kernel_code, std::string &kernel_name)
{
  nvrtcResult rv;
  nvrtcProgram prog;

  // Creating nvrtc program
  rv = nvrtcCreateProgram(
      &prog,
      kernel_code.c_str(),
      nullptr,
      0,
      nullptr,
      nullptr);

  if (rv != NVRTC_SUCCESS)
  {
    return fail_nvrtc(env, rv, "nvrtcCreateProgram");
  }

  // Do this work on Windows? Is it really necessary?
  // - Henrique
  const char *options[10] = {
      "--include-path=/lib/erlang/usr/include/",
      "--include-path=/usr/include/",
      "--include-path=/usr/lib/",
      "--include-path=/usr/include/x86_64-linux-gnu/",
      "--include-path=/usr/include/c++/11",
      "--include-path=/usr/include/x86_64-linux-gnu/c++/11",
      "--include-path=/usr/include/c++/11/backward",
      "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include",
      "--include-path=/usr/include/i386-linux-gnu/",
      "--include-path=/usr/local/include"};

  rv = nvrtcCompileProgram(prog, 10, options);
  if (rv != NVRTC_SUCCESS)
  {
    nvrtcResult erro_g = rv;
    size_t log_size;

    rv = nvrtcGetProgramLogSize(prog, &log_size);
    if (rv != NVRTC_SUCCESS)
    {
      return fail_nvrtc(env, rv, "nvrtcGetProgramLogSize");
    }

    std::string log(log_size, '\0');
    rv = nvrtcGetProgramLog(prog, log.data());

    if (rv != NVRTC_SUCCESS)
    {
      return fail_nvrtc(env, rv, "nvrtcGetProgramLog");
    }

    std::cerr << "NVRTC Compilation failed:\n"
              << log << std::endl;

    return fail_nvrtc(env, erro_g, "nvrtcCompileProgram");
  }

  // Get compiled ptx code
  size_t ptx_size;

  rv = nvrtcGetPTXSize(prog, &ptx_size);
  if (rv != NVRTC_SUCCESS)
  {
    return fail_nvrtc(env, rv, "nvrtcGetPTXSize");
  }

  // Allocate memory for the PTX in heap
  char *ptx_code = new char[ptx_size];

  // Get the PTX code for the compiled program in the allocated memory
  rv = nvrtcGetPTX(prog, ptx_code);
  if (rv != NVRTC_SUCCESS)
  {
    return fail_nvrtc(env, rv, "nvrtcGetPTX");
  }

  // Allocate memory in Erlang for the CompiledKernel resource
  CompiledKernel *compiled_kernel = (CompiledKernel *)enif_alloc_resource(COMPILED_PTX, sizeof(CompiledKernel));

  // Save the compiled PTX code pointer and kernel name in the CompiledKernel resource
  compiled_kernel->ptx = ptx_code;
  compiled_kernel->kernel_name = kernel_name;

  // Destroy the nvrtc program to free resources (we don't need it anymore since we have the PTX code)
  nvrtcDestroyProgram(&prog);

  // Create an Erlang resource for the PTX code and release the allocated resource (the PTX code will be managed by BEAM now)
  ERL_NIF_TERM compiled_kernel_term = enif_make_resource(env, compiled_kernel);
  enif_release_resource(compiled_kernel);

  return compiled_kernel_term;
}

// This function compiles the given kernel code and returns the kernel as an Erlang resource
// Parameters:
// 1 - Kernel name as a charlist
// 2 - Kernel code as a charlist
// Returns:
// - On success: An Erlang resource containing the compiled kernel
static ERL_NIF_TERM jit_compile_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  // Check argc
  if (argc != 2)
  {
    return enif_make_badarg(env);
  }

  // Get kernel name
  ERL_NIF_TERM e_name = argv[0];
  unsigned int size_name;
  if (!enif_get_list_length(env, e_name, &size_name))
  {
    return enif_make_badarg(env);
  }

  std::string kernel_name(size_name, '\0');
  enif_get_string(env, e_name, kernel_name.data(), size_name + 1, ERL_NIF_LATIN1);

  // Get kernel code to compile
  ERL_NIF_TERM e_code = argv[1];
  unsigned int size_code;
  if (!enif_get_list_length(env, e_code, &size_code))
  {
    return enif_make_badarg(env);
  }

  std::string code(size_code, '\0');
  enif_get_string(env, e_code, code.data(), size_code + 1, ERL_NIF_LATIN1);

  // Compile the kernel code to PTX and return the result as an Erlang resource
  return compile_to_ptx(env, code, kernel_name);
}

// Launch a previously compiled kernel with the specified blocks, threads, and arguments.
// Parameters:
// 1 - Kernel Erlang resource (compiled kernel)
// 2 - Blocks as a tuple of three integers (x, y, z)
// 3 - Threads as a tuple of three integers (x, y, z)
// 4 - Number of arguments
// 5 - Types of arguments
// 6 - Arguments
static ERL_NIF_TERM jit_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  // Check argc
  if (argc != 6)
  {
    return enif_make_badarg(env);
  }

  // Getting compiled kernel from the first argument (Erlang resource)
  CompiledKernel *compiled_kernel = NULL;
  if (!enif_get_resource(env, argv[0], COMPILED_PTX, (void **)&compiled_kernel))
  {
    return enif_make_badarg(env);
  }

  // Getting blocks and threads tuples pointers
  const ERL_NIF_TERM *tuple_blocks, *tuple_threads;
  int arity;

  if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks))
  {
    std::cerr << "[ERROR] The given blocks argument is not a tuple." << std::endl;
    return enif_make_badarg(env);
  }
  if (arity != 3)
  {
    std::cerr << "[ERROR] The blocks tuples must have exactly 3 elements (for x, y, z dimensions)." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads))
  {
    std::cerr << "[ERROR] The given threads argument is not a tuple." << std::endl;
    return enif_make_badarg(env);
  }
  if (arity != 3)
  {
    std::cerr << "[ERROR] The threads tuples must have exactly 3 elements (for x, y, z dimensions)." << std::endl;
    return enif_make_badarg(env);
  }

  // Extracting the number of blocks and threads from the tuples
  uint blocks[3], threads[3];

  for (int i = 0; i < 3; i++)
  {
    enif_get_uint(env, tuple_blocks[i], blocks + i);
    enif_get_uint(env, tuple_threads[i], threads + i);
  }

  // Get number of arguments
  int size_args_int;
  if (!enif_get_int(env, argv[3], &size_args_int))
  {
    return enif_make_badarg(env);
  }
  size_t size_args = static_cast<size_t>(size_args_int);

  // --- Getting Arguments ---

  CUdeviceptr *arrays = new CUdeviceptr[size_args];
  float *floats = new float[size_args];
  int *ints = new int[size_args];
  double *doubles = new double[size_args];

  uint arrays_ptr_idx = 0;
  uint floats_ptr_idx = 0;
  uint doubles_ptr_idx = 0;
  uint ints_ptr_idx = 0;

  void **args = new void *[size_args];

  ERL_NIF_TERM list_types;
  ERL_NIF_TERM head_types;
  ERL_NIF_TERM tail_types;

  ERL_NIF_TERM list_args;
  ERL_NIF_TERM head_args;
  ERL_NIF_TERM tail_args;

  list_types = argv[4];
  list_args = argv[5];

  for (int i = 0; i < size_args; i++)
  {
    if (!enif_get_list_cell(env, list_types, &head_types, &tail_types))
    {
      std::cerr << "[ERROR] Failed to get list cell for argument types." << std::endl;
      return enif_make_badarg(env);
    }

    // Get the type name from the head of the types list
    char type_name[1024];
    unsigned int size_type;
    if (!enif_get_list_length(env, head_types, &size_type))
    {
      std::cerr << "[ERROR] Failed to get list length for argument type." << std::endl;
      return enif_make_badarg(env);
    }
    enif_get_string(env, head_types, type_name, size_type + 1, ERL_NIF_LATIN1);

    // Getting actual argument in the head of the arguments list
    if (!enif_get_list_cell(env, list_args, &head_args, &tail_args))
    {
      std::cerr << "[ERROR] Failed to get list cell for argument values." << std::endl;
      return enif_make_badarg(env);
    }

    if (strcmp(type_name, "int") == 0)
    {
      int iarg;
      if (!enif_get_int(env, head_args, &iarg))
      {
        std::cerr << "[ERROR] Failed to get int argument value." << std::endl;
        return enif_make_badarg(env);
      }
      ints[ints_ptr_idx] = iarg;
      args[i] = (void *)&ints[ints_ptr_idx];
      ints_ptr_idx++;
    }
    else if (strcmp(type_name, "float") == 0)
    {
      double darg;
      if (!enif_get_double(env, head_args, &darg))
      {
        std::cerr << "[ERROR] Failed to get float argument value." << std::endl;
        return enif_make_badarg(env);
      }
      floats[floats_ptr_idx] = static_cast<float>(darg);
      args[i] = (void *)&floats[floats_ptr_idx];
      floats_ptr_idx++;
    }
    else if (strcmp(type_name, "double") == 0)
    {
      double darg;
      if (!enif_get_double(env, head_args, &darg))
      {
        std::cerr << "[ERROR] Failed to get double argument value." << std::endl;
        return enif_make_badarg(env);
      }
      doubles[doubles_ptr_idx] = darg;
      args[i] = (void *)&doubles[doubles_ptr_idx];
      doubles_ptr_idx++;
    }
    else if (strcmp(type_name, "tint") == 0 ||
             strcmp(type_name, "tfloat") == 0 ||
             strcmp(type_name, "tdouble") == 0)
    {
      CUdeviceptr *array_res;
      if (!enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res))
      {
        std::cerr << "[ERROR] Failed to get GPU array resource." << std::endl;
        return enif_make_badarg(env);
      }
      arrays[arrays_ptr_idx] = *array_res;
      args[i] = (void *)&arrays[arrays_ptr_idx];
      arrays_ptr_idx++;
    }
    else
    {
      std::cerr << "[ERROR] Type '" << type_name << "' not supported." << std::endl;
      return enif_make_badarg(env);
    }

    list_types = tail_types;
    list_args = tail_args;
  }

  // Launch kernel
  CUresult err;

  CUmodule module;
  err = cuModuleLoadDataEx(&module, compiled_kernel->ptx, 0, 0, 0);
  if (err != CUDA_SUCCESS)
  {
    return fail_cuda(env, err, "cuModuleLoadData jit compile");
  }

  // Get the kernel function from the module using the kernel name
  CUfunction function;
  err = cuModuleGetFunction(&function, module, compiled_kernel->kernel_name.c_str());
  if (err != CUDA_SUCCESS)
  {
    return fail_cuda(env, err, "cuModuleGetFunction jit compile");
  }

  err = cuLaunchKernel(
      function,
      blocks[0], blocks[1], blocks[2],
      threads[0], threads[1], threads[2],
      0, 0, args, 0);
  if (err != CUDA_SUCCESS)
  {
    return fail_cuda(env, err, "cuLaunchKernel jit compile");
  }

  err = cuCtxSynchronize();
  if (err != CUDA_SUCCESS)
  {
    const char *error;
    cuGetErrorString(err, &error);

    std::string message = "Error (jit_launch_nif): error synchronizing device after kernel launch: " + std::string(error);

    enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  // Clean up allocated memory for arguments
  delete[] arrays;
  delete[] floats;
  delete[] ints;
  delete[] doubles;
  delete[] args;

  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {"new_empty_gpu_array_nif", 3, new_empty_gpu_array_nif, 0},
    {"new_gpu_array_from_nx_nif", 4, new_gpu_array_from_nx_nif, 0},
    {"get_gpu_array_nif", 4, get_gpu_array_nif, 0},
    {"synchronize_nif", 0, synchronize_nif, 0},
    {"jit_compile_nif", 2, jit_compile_nif, 0},
    {"jit_launch_nif", 6, jit_launch_nif, 0}};

ERL_NIF_INIT(Elixir.CudaBackend, nif_funcs, &load, NULL, NULL, &unload)