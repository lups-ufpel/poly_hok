/*
    This file implements the Native Implemented Functions (NIFs) for GPU operations using OpenCL
    in Elixir.

    Ported to OpenCL/C++ by: Henrique Gabriel Rodrigues
    Oriented and supervised by: Prof. Dr. André Rauber Du Bois
    Original code by: Prof. Dr. André Rauber Du Bois
*/

#include "ocl_interface/OCLInterface.hpp"

#include <erl_nif.h>

#include <iostream>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

bool debug_logs = false;

bool gpu_double_support = false;

OCLInterface *open_cl = nullptr;

// Global resource type for GPU arrays (cl::Buffer objects)
ErlNifResourceType *ARRAY_TYPE;
// Global resource for compiled kernels
ErlNifResourceType *KERNEL_TYPE;

// Destructor for device array resource (cl::Buffer)
void dev_array_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Buffer *dev_array = (cl::Buffer *)res;

  // Explicitly call the destructor for the cl::Buffer object without deallocating
  // the resource memory itself (the memory where the pointer to cl::Buffer is stored).
  // This is Erlang's garbage collector responsibility, and if we do this we'll get a
  // deallocation error.
  dev_array->~Buffer();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Device array resource destroyed." << std::endl;
  }
}

// Destructor for compiled kernel resources (cl::Kernel)
void kernel_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Kernel *kernel = (cl::Kernel *)res;

  // Explicitly call the destructor for the cl::Kernel object without deallocating
  // the resource memory itself (the memory where the pointer to cl::Kernel is stored).
  // This is Erlang's garbage collector responsibility, and if we do this we'll get a
  // deallocation error.
  kernel->~Kernel();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Kernel resource destroyed." << std::endl;
  }
}

// This function initializes the OpenCL interface, selects the default platform, GPU device,
// and checks for required extension support.
void init_ocl(ErlNifEnv *env)
{
  if (open_cl != nullptr)
    return; // Already initialized

  open_cl = new OCLInterface(debug_logs);

  try
  {
    // Selecting default platform and device
    open_cl->selectDefaultPlatformAndDevice(CL_DEVICE_TYPE_GPU);

    // Check for extension support
    std::vector<std::string> desired_extensions = {"cl_khr_fp64", "cl_khr_int64_base_atomics"};
    std::vector<std::pair<std::string, bool>> extensions_support = open_cl->checkDeviceExtensions(desired_extensions);

    bool fp64_supported = false;
    bool int64_base_atomics_supported = false;

    // Update global flags for extension support
    for (const auto &ext : extensions_support)
    {
      if (ext.first == "cl_khr_fp64")
      {
        fp64_supported = ext.second;
      }
      else if (ext.first == "cl_khr_int64_base_atomics")
      {
        int64_base_atomics_supported = ext.second;
      }
    }

    // If both extensions are supported, then this device can handle the double type
    if (fp64_supported && int64_base_atomics_supported)
    {
      gpu_double_support = true;

      // Define flag for double support in OpenCL build options
      open_cl->setBuildOptions("-D DOUBLE_SUPPORTED=1");
    }

    // Add ignore warnings build option
    open_cl->setBuildOptions(open_cl->getBuildOptions() + " -w");
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Failed to initialize OpenCL interface: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));

    delete open_cl;
  }
}

// This function is called when this NIF library is loaded
static int load(ErlNifEnv *env, void ** /* priv_data */, ERL_NIF_TERM /* load_info */)
{
  // Defines the Erlang resource type for GPU arrays (Buffer objects in our case)
  ARRAY_TYPE = enif_open_resource_type(
      env,
      NULL,
      "gpu_ref",
      dev_array_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Defines the Erlang resource for compiled kernels (cl::Kernel)
  KERNEL_TYPE = enif_open_resource_type(
      env,
      NULL,
      "kernel_ref",
      kernel_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Initialize OpenCL interface
  init_ocl(env);

  return 0;
}

// This function is called when the NIF library is unloaded
static void unload(ErlNifEnv * /* env */, void * /* priv_data */)
{
  if (open_cl != nullptr)
  {
    delete open_cl;
    open_cl = nullptr;
  }

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] GPU NIFs unloaded successfully." << std::endl;
  }
}

// This function retrieves the OpenCL array from the device (GPU) and returns it to the host as an Erlang term.
static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;
  char type_name[1024];

  cl::Buffer *device_array = nullptr;

  // Get the Buffer resource to copy data from
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&device_array))
  {
    return enif_make_badarg(env);
  }

  // Get number of rows
  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
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

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // Calculating the size of the result
  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * nrow * ncol;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * nrow * ncol;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * nrow * ncol;
  }
  else // Unknown type
  {
    char message[200];
    snprintf(message, sizeof(message),
             "[ERROR] (get_gpu_array_nif) copying data from device to host: unknown type %s",
             type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Allocate memory in the host to store the result
  // According to Erlang's docs, for LARGE binaries, it is better to use enif_alloc_binary.
  ErlNifBinary host_bin;

  if (!enif_alloc_binary(data_size, &host_bin))
  {
    char message[200];
    snprintf(message, sizeof(message),
             "[ERROR] (get_gpu_array_nif) failed to allocate binary of size %zu",
             data_size);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Getting pointer to the allocated binary data in the host
  void *result_data_pointer = (void *)host_bin.data;

  // Copying data from device to host
  try
  {
    open_cl->readBuffer(*device_array, result_data_pointer, data_size);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Data copied from device to host successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (get_gpu_array_nif) copying data from device to host: " << e.what() << std::endl;
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  // Creating the Erlang binary term to return (passing ownership to the BEAM)
  ERL_NIF_TERM erl_term_result = enif_make_binary(env, &host_bin);
  return erl_term_result;
}

// This function creates a new GPU array with the specified number of rows, columns, and type.
// It allocates memory on the GPU and copies data to it from the host array provided.
static ERL_NIF_TERM new_gpu_array_from_nx_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;
  ErlNifBinary host_array_el;

  // Get the host array binary
  if (!enif_inspect_binary(env, argv[0], &host_array_el))
  {
    return enif_make_badarg(env);
  }

  // Get number of rows
  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
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
  else // Unknown type
  {
    char message[200];
    snprintf(message, sizeof(message),
             "[ERROR] (new_gpu_array_from_nx_nif): unknown type %s",
             type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  try
  {
    // Allocate an empty buffer on the GPU for the array
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE);
    // Copy data from host to device (H2D copy)
    open_cl->writeBuffer(dev_array, (void *)host_array_el.data, data_size);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (gpu_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(gpu_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New GPU array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
      std::cout << "[C++ GPU NIF] Data copied from host to device successfully." << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

// Creates a new empty GPU array with the specified number of rows, columns, and type
static ERL_NIF_TERM new_empty_gpu_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;

  // Get number of rows
  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
  if (!enif_get_int(env, argv[1], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  // The type name is a list of characters, so we need to get its length first
  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  // Create a buffer to hold the type name
  // We add 1 to the size to accommodate the null terminator
  // Note: ERL_NIF_LATIN1 is used for encoding the string
  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // From here on, we will use the type name to determine the data size and allocate memory accordingly
  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * nrow * ncol;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * nrow * ncol;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * nrow * ncol;
  }
  else // Unknown type
  {
    char message[200];
    snprintf(message, sizeof(message),
             "[ERROR] new_empty_gpu_array_nif: unknown type: %s",
             type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  try
  {
    // Allocate memory on the GPU
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (gpu_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(gpu_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New GPU array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

// This function synchronizes the OpenCL command queue, ensuring that all previously enqueued commands have completed.
static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM /* argv */[])
{
  open_cl->synchronize();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] OpenCL command queue synchronized successfully." << std::endl;
  }

  return enif_make_int(env, 0);
}

// This function sets the debug logs flag for the NIFs.
static ERL_NIF_TERM set_debug_logs_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 1)
  {
    std::cerr << "[ERROR] Invalid number of arguments for set_debug_logs_nif." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_is_atom(env, argv[0]))
  {
    std::cerr << "[ERROR] Argument for set_debug_logs_nif must be either 'true' or 'false' atoms." << std::endl;
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM true_atom = enif_make_atom(env, "true");

  debug_logs = (enif_compare(argv[0], true_atom) == 0);
  open_cl->setDebugLogs(debug_logs);

  return enif_make_int(env, 0);
}

// This function checks if the current device supports double precision floating points
// and int64 base atomics extensions for CAS
static ERL_NIF_TERM double_supported_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM /* argv */[])
{
  if (gpu_double_support)
  {
    return enif_make_atom(env, "true");
  }
  else
  {
    return enif_make_atom(env, "false");
  }
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
    std::cerr << "[ERROR] Invalid number of arguments for jit_compile_nif." << std::endl;
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

  // Allocating memory inside BEAM to hold the cl::Kernel object.
  void *raw_memory = enif_alloc_resource(KERNEL_TYPE, sizeof(cl::Kernel));
  // Creating the cl::Kernel object in the allocated memory using placement new.
  cl::Kernel *kernel = new (raw_memory) cl::Kernel();

  // Create and compile kernel program
  try
  {
    cl::Program program = open_cl->createProgram(code);
    *kernel = open_cl->createKernel(program, kernel_name.c_str());
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  // Create resource and release it, so that BEAM can manage its lifetime.
  ERL_NIF_TERM kernel_resource = enif_make_resource(env, kernel);
  enif_release_resource(kernel);

  return kernel_resource;
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
    std::cerr << "[ERROR] Invalid number of arguments for jit_launch_nif." << std::endl;
    return enif_make_badarg(env);
  }

  // Getting kernel resource from the first argument
  cl::Kernel *kernel = nullptr;
  if (!enif_get_resource(env, argv[0], KERNEL_TYPE, (void **)&kernel))
  {
    return enif_make_badarg(env);
  }

  // Reading kernel name
  std::string kernel_name = kernel->getInfo<CL_KERNEL_FUNCTION_NAME>();

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
  int blocks[3], threads[3];

  for (int i = 0; i < 3; i++)
  {
    enif_get_int(env, tuple_blocks[i], blocks + i);
    enif_get_int(env, tuple_threads[i], threads + i);
  }

  // Creating NDRange objects for local and global range
  cl::NDRange global_range, local_range;

  // If the user wants OpenCL to calculate the number of threads automatically, Elixir will set the threads tuple to {0, 0, 0}.
  // This is the only case where the threads tuple can contain zero, so we can check only if the first element is zero.
  bool let_opencl_decide_local_range = (threads[0] == 0);

  if (let_opencl_decide_local_range)
  {
    // Let OpenCL decide the local range (work-group size)
    local_range = cl::NullRange;
    // In this case, the grid size will have to contain the global range
    global_range = cl::NDRange(blocks[0], blocks[1], blocks[2]);
  }
  else
  {
    local_range = cl::NDRange(threads[0], threads[1], threads[2]);
    global_range = cl::NDRange(blocks[0] * threads[0], blocks[1] * threads[1], blocks[2] * threads[2]);
  }

  if (debug_logs)
  {
    if (let_opencl_decide_local_range)
    {
      std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' will be executed with a global range of "
                << global_range[0] << "x" << global_range[1] << "x" << global_range[2]
                << " and an automatically determined local range." << std::endl;
    }
    else
    {
      std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' will be executed with a global range of "
                << global_range[0] << "x" << global_range[1] << "x" << global_range[2]
                << " and a local range of " << local_range[0] << "x" << local_range[1]
                << "x" << local_range[2] << "." << std::endl;
    }
  }

  // Getting the number of arguments given to the kernel
  int size_args;
  if (!enif_get_int(env, argv[3], &size_args))
  {
    return enif_make_badarg(env);
  }

  // Collecting the arguments and their types
  ERL_NIF_TERM list_args_types;
  ERL_NIF_TERM head_args_types;
  ERL_NIF_TERM tail_args_types;

  ERL_NIF_TERM list_args;
  ERL_NIF_TERM head_args;
  ERL_NIF_TERM tail_args;

  list_args_types = argv[4];
  list_args = argv[5];

  for (int i = 0; i < size_args; i++)
  {
    ERL_NIF_TERM arg;
    char arg_type_name[1024];
    unsigned int arg_type_name_lenght;

    // Get first element of the list of types
    if (!enif_get_list_cell(env, list_args_types, &head_args_types, &tail_args_types))
    {
      std::cerr << "[ERROR] Error getting list cell for kernel argument types." << std::endl;
      return enif_make_badarg(env);
    }

    // Get length of the type name
    if (!enif_get_list_length(env, head_args_types, &arg_type_name_lenght))
    {
      std::cerr << "[ERROR] Error getting type name length for kernel argument types." << std::endl;
      return enif_make_badarg(env);
    }

    // Get the type name as a string
    enif_get_string(env, head_args_types, arg_type_name, arg_type_name_lenght + 1, ERL_NIF_LATIN1);

    // Get first element of the list of arguments
    // This is the actual argument that will be passed to the kernel
    if (!enif_get_list_cell(env, list_args, &head_args, &tail_args))
    {
      std::cerr << "[ERROR] Error getting list cell for kernel argument " << i << "." << std::endl;
      return enif_make_badarg(env);
    }
    arg = head_args;

    // Now that we have the argument and its type name
    // We can convert the argument to the appropriate type and set it in the kernel object
    if (strcmp(arg_type_name, "int") == 0)
    {
      int iarg;
      if (!enif_get_int(env, arg, &iarg))
      {
        std::cerr << "[ERROR] Error getting integer argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel->setArg(i, iarg);
    }
    else if (strcmp(arg_type_name, "float") == 0)
    {
      double darg;
      if (!enif_get_double(env, arg, &darg))
      {
        std::cerr << "[ERROR] Error getting float argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      float farg = static_cast<float>(darg);
      kernel->setArg(i, farg);
    }
    else if (strcmp(arg_type_name, "double") == 0)
    {
      double darg;
      if (!enif_get_double(env, arg, &darg))
      {
        std::cerr << "[ERROR] Error getting double argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel->setArg(i, darg);
    }
    else if (
        strcmp(arg_type_name, "tint") == 0 ||
        strcmp(arg_type_name, "tfloat") == 0 ||
        strcmp(arg_type_name, "tdouble") == 0)
    {
      cl::Buffer *array_res;
      if (!enif_get_resource(env, arg, ARRAY_TYPE, (void **)&array_res))
      {
        std::cerr << "[ERROR] Error getting buffer (array) resource for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel->setArg(i, *array_res);
    }
    else
    {
      std::cerr << "[ERROR] Unknown argument type '" << arg_type_name << "' for kernel." << std::endl;
      return enif_make_badarg(env);
    }

    list_args_types = tail_args_types;
    list_args = tail_args;
  }

  // Now we can execute the kernel
  try
  {
    open_cl->executeKernel(*kernel, global_range, local_range);
    open_cl->synchronize();

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' executed successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {"new_empty_gpu_array_nif", 3, new_empty_gpu_array_nif, 0},
    {"new_gpu_array_from_nx_nif", 4, new_gpu_array_from_nx_nif, 0},
    {"get_gpu_array_nif", 4, get_gpu_array_nif, 0},
    {"synchronize_nif", 0, synchronize_nif, 0},
    {"set_debug_logs_nif", 1, set_debug_logs_nif, 0},
    {"double_supported_nif", 0, double_supported_nif, 0},
    {"jit_compile_nif", 2, jit_compile_nif, 0},
    {"jit_launch_nif", 6, jit_launch_nif, 0}};

ERL_NIF_INIT(Elixir.OpenclBackend, nif_funcs, &load, NULL, NULL, &unload)