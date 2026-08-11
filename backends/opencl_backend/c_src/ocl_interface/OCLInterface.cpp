#include "OCLInterface.hpp"

OCLInterface::OCLInterface() : OCLInterface(false) {}

OCLInterface::OCLInterface(bool enable_debug_logs)
{
    // Initialize OpenCL interface with null pointers for
    // for good error handling and predictable behavior
    this->selected_platform = nullptr;
    this->selected_device = nullptr;
    this->context = nullptr;
    this->command_queue = nullptr;

    this->build_options = "";
    this->debug_logs = enable_debug_logs;
}

OCLInterface::~OCLInterface()
{
    // Clean up OpenCL resources if they were created
    if (this->command_queue() != nullptr)
    {
        this->command_queue.finish();
    }
}

void OCLInterface::setDebugLogs(bool enable)
{
    this->debug_logs = enable;
}

void OCLInterface::createContext()
{
    this->context = cl::Context(this->selected_device);
}

void OCLInterface::createCommandQueue()
{
    this->command_queue = cl::CommandQueue(this->context, this->selected_device);
}

std::string OCLInterface::getKernelCode(const char *file_name)
{
    std::ifstream kernel_file(file_name);
    std::string output, line;

    if (!kernel_file.is_open())
    {
        std::cerr << "[OCL C++ Interface] Unable to open kernel file '" << file_name << "'." << std::endl;
        throw std::runtime_error("Kernel file not found");
    }

    while (std::getline(kernel_file, line))
    {
        output += line += "\n";
    }

    kernel_file.close();

    return output;
}

std::vector<cl::Platform> OCLInterface::getAvailablePlatforms()
{
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    return platforms;
}

void OCLInterface::selectPlatform(cl::Platform p)
{
    if (p() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL platform selected");
    }

    this->selected_platform = p;

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Selected OpenCL platform: " << this->selected_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }
}

void OCLInterface::selectDevice(cl::Device d)
{
    if (d() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL device selected");
    }

    this->selected_device = d;

    this->createContext();
    this->createCommandQueue();

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Selected OpenCL device: " << this->selected_device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
}

void OCLInterface::selectDefaultPlatformAndDevice(cl_device_type device_type)
{
    std::vector<cl::Platform> available_platforms = this->getAvailablePlatforms();
    std::vector<cl::Device> available_devices;

    for (const cl::Platform &p : available_platforms)
    {
        p.getDevices(device_type, &available_devices);

        if (available_devices.empty())
        {
            continue;
        }
        else
        {
            selectPlatform(p);
            selectDevice(available_devices[0]);

            break;
        }
    }
}

std::vector<std::pair<std::string, bool>> OCLInterface::checkDeviceExtensions(std::vector<std::string> &extensions)
{
    if (this->selected_device() == nullptr)
    {
        throw std::runtime_error("No OpenCL device selected");
    }

    std::string device_extensions = this->selected_device.getInfo<CL_DEVICE_EXTENSIONS>();
    std::vector<std::pair<std::string, bool>> results;

    for (const auto &ext : extensions)
    {
        bool supported = (device_extensions.find(ext) != std::string::npos);
        results.push_back(std::make_pair(ext, supported));
    }

    return results;
}

void OCLInterface::setBuildOptions(const std::string &options)
{
    this->build_options = options;

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Set OpenCL build options: " << this->build_options << std::endl;
    }
}

cl::Program OCLInterface::createProgram(std::string &program_code)
{
    cl::Program program(this->context, program_code);

    try
    {
        program.build(this->selected_device, this->build_options.c_str());
    }
    catch (const cl::BuildError &err)
    {
        std::cerr << "[OCL C++ Interface] Build Error!" << std::endl;

        std::string device_name = err.getBuildLog().front().first.getInfo<CL_DEVICE_NAME>();
        std::string build_log = err.getBuildLog().front().second;

        std::cerr << "> Device: " << device_name << std::endl;
        std::cerr << "> Build Log:\n"
                  << std::endl;
        std::cerr << build_log << std::endl;

        throw std::runtime_error("Failed to build OpenCL program");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL program created and builded successfully." << std::endl;
    }

    return program;
}

cl::Program OCLInterface::createProgram(const char *program_code)
{
    std::string code_str(program_code);
    return this->createProgram(code_str);
}

cl::Program OCLInterface::createProgramFromFile(const char *file_name)
{
    std::string code_str = this->getKernelCode(file_name);
    return this->createProgram(code_str);
}

cl::Kernel OCLInterface::createKernel(const cl::Program &program, const char *kernel_name)
{
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program, kernel_name, &err);

    if (err != CL_SUCCESS || kernel() == nullptr)
    {
        std::cerr << "[OCL C++ Interface] Failed to create OpenCL kernel '" << kernel_name << "'. Error code: " << err << std::endl;
        throw std::runtime_error("Failed to create OpenCL kernel");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL kernel '" << kernel_name << "' created successfully." << std::endl;
    }

    return kernel;
}

cl::Buffer OCLInterface::createBuffer(size_t size, cl_mem_flags flags, void *host_ptr)
{
    try
    {
        cl::Buffer buffer(this->context, flags, size, host_ptr);

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] OpenCL buffer of size " << size << " created successfully." << std::endl;
        }

        return buffer;
    }
    catch (const cl::Error &e)
    {
        // e.what() provides a description of the error, for example "clCreateBuffer"
        // e.err() provides the OpenCL error code (e.g., CL_MEM_OBJECT_ALLOCATION_FAILURE)
        cl_int error_code = e.err();
        std::string error_msg;

        // Retrieve device memory info for better error messages
        cl_ulong global_mem = this->selected_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();   // Total global memory size
        cl_ulong max_alloc = this->selected_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(); // Max allocation size allowed

        // Converting sizes to MB for easier readability
        cl_ulong global_mem_mb = global_mem / (1024 * 1024);
        cl_ulong max_alloc_mb = max_alloc / (1024 * 1024);
        cl_ulong buff_size_mb = size / (1024 * 1024);

        switch (error_code)
        {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            std::cerr << "[OCL C++ Interface] Error: GPU out of memory for buffer allocation of size " << size << "." << std::endl;
            std::cerr << "> Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << "> Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = "GPU out of memory";
            break;

        case CL_INVALID_BUFFER_SIZE:
            std::cerr << "[OCL C++ Interface] Error: Invalid buffer size requested: " << size << " bytes." << std::endl;
            std::cerr << ">  Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << ">  Device Max Allocation Size: " << max_alloc_mb << " MB" << std::endl;
            std::cerr << ">  Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = "Invalid buffer size requested";
            break;

        default:
            std::cerr << "[OCL C++ Interface] Failed to create OpenCL buffer of size " << size << "." << std::endl;
            std::cerr << "> Error: " << e.what() << std::endl;
            std::cerr << "> Error code: " << std::to_string(error_code) << std::endl;

            std::cerr << "\n[Device Memory Info]" << std::endl;
            std::cerr << "> Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << "> Device Max Allocation Size: " << max_alloc_mb << " MB" << std::endl;
            std::cerr << "> Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = std::string(e.what()) + " (Error code: " + std::to_string(error_code) + ")";
            break;
        }

        throw std::runtime_error(error_msg);
    }
}

void OCLInterface::executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range)
{
    try
    {
        this->command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
    }
    catch (const cl::Error &err)
    {
        std::cerr << "[OCL C++ Interface] Failed to execute OpenCL kernel." << std::endl;
        std::cerr << "> Error code: " << err.err() << std::endl;
        std::cerr << "> Error message: " << err.what() << std::endl;
        throw std::runtime_error("Failed to execute OpenCL kernel");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL kernel executed successfully." << std::endl;
    }
}

void OCLInterface::readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, size_t offset) const
{
    this->command_queue.enqueueReadBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void OCLInterface::writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, size_t offset) const
{
    // This function could be non-blocking in the future...
    this->command_queue.enqueueWriteBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void *OCLInterface::mapHostPtrToPinnedMemory(const cl::Buffer &buffer, cl_map_flags flags, size_t size, size_t offset) const
{
    cl_int err;

    void *mapped_ptr = this->command_queue.enqueueMapBuffer(buffer, CL_TRUE, flags, offset, size, nullptr, nullptr, &err);

    if (err != CL_SUCCESS)
    {
        std::cerr << "[OCL C++ Interface] Failed to map OpenCL buffer to pinned memory. Error code: " << err << std::endl;
        throw std::runtime_error("Failed to map OpenCL buffer to pinned memory");
    }

    return mapped_ptr;
}

void OCLInterface::unMapHostPtr(const cl::Buffer &buffer, void *host_ptr) const
{
    this->command_queue.enqueueUnmapMemObject(buffer, host_ptr);
}

void OCLInterface::synchronize() const
{
    this->command_queue.finish(); // Wait for all commands up to this point to complete in the command queue
}