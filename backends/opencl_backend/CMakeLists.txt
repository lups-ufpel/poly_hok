cmake_minimum_required(VERSION 3.12)
project(OCL_PolyHok_Runtime VERSION 0.1.1 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ---- Find OpenCL - it is REQUIRED, so CMake will stop if it is not found ----

# This is automatically handled by CMake
find_package(OpenCL REQUIRED)

# ---- Set sources ----

set(GPU_NIFS_SRC
    c_src/gpu_nifs.cpp
)

set(BMP_NIFS_SRC
    c_src/bmp_nifs.cpp
)

# ---- Creating libraries used by the NIFs and making their headers available ----

add_library(OCL_INTERFACE_LIB STATIC
    c_src/ocl_interface/OCLInterface.cpp
)
target_include_directories(OCL_INTERFACE_LIB PUBLIC c_src)
target_link_libraries(OCL_INTERFACE_LIB PUBLIC OpenCL::OpenCL)

add_library(BMP_LIB STATIC
    c_src/bmp/BMP.cpp
)
target_include_directories(BMP_LIB PUBLIC c_src)

# Marking the libraries as POSITION_INDEPENDENT_CODE for shared library compatibility
set_target_properties(OCL_INTERFACE_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(BMP_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Creating the NIF libraries and linking them with the appropriate libraries ----
add_library(gpu_nifs SHARED ${GPU_NIFS_SRC})
target_link_libraries(gpu_nifs PRIVATE OCL_INTERFACE_LIB)
target_include_directories(gpu_nifs PRIVATE ${ERLANG_INCLUDE_DIR})

add_library(bmp_nifs SHARED ${BMP_NIFS_SRC})
target_link_libraries(bmp_nifs PRIVATE BMP_LIB)
target_include_directories(bmp_nifs PRIVATE ${ERLANG_INCLUDE_DIR})

# Removing the 'lib' prefix from the generated shared libraries
set_target_properties(gpu_nifs PROPERTIES PREFIX "")
set_target_properties(bmp_nifs PROPERTIES PREFIX "")

# ---- Copy the generated shared libraries to the 'priv' folder after building ----
add_custom_command(TARGET gpu_nifs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/priv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:gpu_nifs>" "${CMAKE_SOURCE_DIR}/priv/$<TARGET_FILE_NAME:gpu_nifs>"
    COMMENT "Copying gpu_nifs to priv/ directory..."
)

add_custom_command(TARGET bmp_nifs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/priv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:bmp_nifs>" "${CMAKE_SOURCE_DIR}/priv/$<TARGET_FILE_NAME:bmp_nifs>"
    COMMENT "Copying bmp_nifs to priv/ directory..."
)