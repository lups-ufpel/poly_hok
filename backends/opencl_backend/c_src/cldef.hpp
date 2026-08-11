/*
    This header defines the OpenCL version to use, include OpenCL headers and enables C++ exceptions.
    
    It is configured to use OpenCL 3.0 because, believe it or not, it is more adopted than OpenCL 2.0

    Made by: Henrique Gabriel Rodrigues, Eduardo Mailan
    Supervised by: Prof. Dr. André Rauber Du Bois

    © Laboratory of Ubiquitous and Parallel Systems (LUPS)
*/

#pragma once

#define OPENCL_VERSION 300 // We are going to be using OpenCL 2.0

#define CL_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_ENABLE_EXCEPTIONS 1

#include <CL/opencl.hpp>