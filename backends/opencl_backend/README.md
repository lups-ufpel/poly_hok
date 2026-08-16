# OpenclBackend

This is the OpenCL backend for PolyHok. It is responsible for generating OpenCL code from PolyHok kernels and device functions ASTs, compiling and executing this generated OpenCL code, and managing the OpenCL context and resources.

It implements the `BackendBehavior` defined in the core module, providing the necessary functionality for PolyHok to run on OpenCL-compatible devices.

This module defines NIFs for interfacing with the OpenCL runtime in the host environment. The NIFs are implemented in C++ and compiled with CMake using the `cmake_compiler` module. The NIFs code are located in the `c_src/` directory, and the CMake build configuration is in the `CMakeLists.txt` file.
