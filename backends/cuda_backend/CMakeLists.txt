cmake_minimum_required(VERSION 3.12)
project(CUDA_PolyHok_Runtime VERSION 0.1.1 LANGUAGES CXX CUDA)

# Set standard to C++17 for both C++ and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake needs to know which GPU architecture(s) to generate code for.
# "native" (CMake >= 3.24) targets whatever GPU is present on the build
# machine, which mirrors nvcc's default behavior closely enough for a dev
# build. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> if you want to
# build for a different architecture (e.g., 75 for Turing, 86 for Ampere).
if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

# ---- Find Erlang development libraries - it is REQUIRED, we can't proceed without this!! ----

# Get erlang executable
find_program(ERLANG_EXECUTABLE erl)

if(NOT ERLANG_EXECUTABLE)
    message(FATAL_ERROR "Erlang executable (erl) not found! Please make sure you have Erlang and Elixir installed and their executables are in your PATH.")
endif()

# Get Erlang include directory by running a command in the shell and capturing its output
execute_process(
    COMMAND ${ERLANG_EXECUTABLE} -noshell -eval "io:format(\"~s\", [lists:concat([code:root_dir(), \"/erts-\", erlang:system_info(version), \"/include\"])]), halt()."
    OUTPUT_VARIABLE ERLANG_INCLUDE_DIR
    RESULT_VARIABLE ERLANG_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the command executed successfully and if the expected header file exists in the include directory
if(NOT ERLANG_RESULT EQUAL 0 OR NOT EXISTS "${ERLANG_INCLUDE_DIR}/erl_nif.h")
    message(FATAL_ERROR "Could not find erl_nif.h inside ${ERLANG_INCLUDE_DIR}")
endif()

# If everything works, this will print the path to the include directory where erl_nif.h is located
message(STATUS "Found Erlang include directory: ${ERLANG_INCLUDE_DIR}")

# ---- Find CUDA - it is REQUIRED, so CMake will stop if it is not found ----

# Gives us the CUDA::cuda_driver and CUDA::nvrtc imported targets used below,
# equivalent to the Makefile's -lcuda -lnvrtc.
find_package(CUDAToolkit REQUIRED)

# ---- Set sources ----

set(GPU_NIFS_SRC
    c_src/gpu_nifs.cu
)

# ---- Creating the NIF library and linking it with CUDA driver API and NVRTC ----

add_library(gpu_nifs SHARED ${GPU_NIFS_SRC})
target_include_directories(gpu_nifs PRIVATE ${ERLANG_INCLUDE_DIR})
target_link_libraries(gpu_nifs PRIVATE CUDA::cuda_driver CUDA::nvrtc)

# Matches the Makefile's --compiler-options '-fPIC' (CMake already defaults
# this on for SHARED targets, set explicitly for clarity).
set_target_properties(gpu_nifs PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Removing the 'lib' prefix from the generated shared library
set_target_properties(gpu_nifs PROPERTIES PREFIX "")

# ---- Copy the generated shared library to the 'priv' folder after building ----
add_custom_command(TARGET gpu_nifs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/priv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:gpu_nifs>" "${CMAKE_SOURCE_DIR}/priv/$<TARGET_FILE_NAME:gpu_nifs>"
    COMMENT "Copying gpu_nifs to priv/ directory..."
)