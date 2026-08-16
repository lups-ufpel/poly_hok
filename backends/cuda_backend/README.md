# CudaBackend

This is the CUDA backend for PolyHok. It is responsible for generating CUDA code from PolyHok kernels and device functions ASTs, compiling and executing this generated CUDA code, and managing the CUDA context and resources.

It implements the `BackendBehavior` defined in the core module, providing the necessary functionality for PolyHok to run on CUDA-compatible devices.

This module defines NIFs for interfacing with the CUDA runtime in the host environment. The NIFs are implemented in C++/CUDA and compiled with CMake using the `cmake_compiler` module. The NIFs code are located in the `c_src/` directory, and the CMake build configuration is in the `CMakeLists.txt` file.

To compile CUDA code in runtime, we use the `nvrtc` library, which is a runtime compilation library provided by NVIDIA. To manage the context, allocate/deallocate GPU memory, and manage other CUDA resources, we used the CUDA Driver API.
