#pragma once

#include "../cldef.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

/**
 * @class OCLInterface
 * @brief A class that provides an interface for OpenCL operations, including platform and device selection,
 * context and command queue creation, program and kernel management, and buffer operations.
 * 
 * This class simplifies the process of working with OpenCL by encapsulating common tasks.
 * 
 * @authors Henrique Gabriel Rodrigues, Prof. Dr. André Rauber Du Bois
 */
class OCLInterface
{
private:
    cl::Platform selected_platform;
    cl::Device selected_device;
    cl::Context context;

    cl::CommandQueue command_queue;

    /**
     * @brief Creates the OpenCL context for the selected_device.
     */
    void createContext();
    /**
     * @brief Creates the OpenCL command queue for the selected_device and context.
     */
    void createCommandQueue();

    /**
     * @brief Returns a vector of available OpenCL platforms on the system.
     * 
     * @return A vector containing the available OpenCL platforms.
     */
    std::vector<cl::Platform> getAvailablePlatforms();
    /**
     * @brief Selects the given OpenCL platform as the active platform.
     * 
     * @param p The OpenCL platform to select.
     */
    void selectPlatform(cl::Platform p);

    /**
     * @brief Selects the given OpenCL device as the active device.
     * 
     * @param d The OpenCL device to select.
     */
    void selectDevice(cl::Device d);

    /**
     * @brief Reads the kernel code from a file.
     * 
     * @param file_name The name of the file containing the kernel code.
     * @return The kernel code as a string.
     */
    std::string getKernelCode(const char *file_name);

    /**
     * @brief OpenCL program build options.
     */
    std::string build_options;
    /**
     * @brief Debug logs flag.
     */
    bool debug_logs;

public:
    /**
     * @brief Default constructor for OCLInterface.
     *
     * Initializes an OCLInterface object with default settings.
     * Delegates construction to the parameterized constructor with 'false' as the argument for enable_debug_logs.
     */
    OCLInterface();
    /**
     * @brief Parameterized constructor for OCLInterface.
     *
     * Initializes an OCLInterface object with the specified debug log setting. The selected platform, device,
     * context, and command queue are initialized to null pointers for better error handling and predictable behavior.
     *
     * @param enable_debug_logs A boolean flag to enable or disable debug logs.
     */
    OCLInterface(bool enable_debug_logs);
    /**
     * @brief Destructor for OCLInterface.
     *
     * Guarantees that all operations in the command queue are completed before the object is destroyed,
     * ensuring proper cleanup of OpenCL resources.
     */
    ~OCLInterface();

    /**
     * @brief Sets the debug logs flag.
     *
     * @param enable A boolean flag to enable or disable debug logs.
     */
    void setDebugLogs(bool enable);

    /**
     * @brief Selects the first available OpenCL platform that contains a device of the specified type, 
     * and selects that device as the active device. If no device type is specified, all device types are considered.
     * 
     * @param device_type The type of OpenCL device to select (default is CL_DEVICE_TYPE_ALL).
     */
    void selectDefaultPlatformAndDevice(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

    /**
     * @brief Checks if the selected device supports the specified OpenCL extensions.
     * 
     * @param extensions A vector of strings containing extension names to check for support. Example: {"cl_khr_fp64"}
     * @return A vector of pairs, where each pair contains the extension name and a boolean indicating support (true if  
     * the extension is supported, false otherwise).
     */
    std::vector<std::pair<std::string, bool>> checkDeviceExtensions(std::vector<std::string> &extensions);
    /**
     * @brief Sets the OpenCL program build options.
     * 
     * @param options A string containing the build options. Example: "-D MY_DEFINE=1 -cl-fast-relaxed-math"
     */
    void setBuildOptions(const std::string &options);
    
    /**
     * @brief Creates and builds an OpenCL program from the given program code string. The build options are used during the build process.
     * If something goes wrong during the build, a detailed error message is printed to stderr and a runtime_error exception is thrown.
     * 
     * @param program_code A string containing the OpenCL program source code.
     * @return The created OpenCL program.
     */
    cl::Program createProgram(std::string &program_code);
    /**
     * @brief Creates an OpenCL program from the given program code C-string. Same behavior as createProgram(std::string &).
     * 
     * @param program_code A C-string containing the OpenCL program source code.
     * @return The created OpenCL program.
     */
    cl::Program createProgram(const char *program_code);
    /**
     * @brief Creates an OpenCL program by reading the kernel code from the specified file. Same behavior as createProgram(std::string &).
     * 
     * @param file_name The name of the file containing the OpenCL kernel code.
     * @return The created OpenCL program.
     */
    cl::Program createProgramFromFile(const char *file_name);
    
    /**
     * @brief Creates an OpenCL kernel from the given program and kernel name.
     * 
     * @param program The OpenCL program containing the kernel.
     * @param kernel_name The name of the kernel to create as a C-string.
     * @return The created OpenCL kernel.
     */
    cl::Kernel createKernel(const cl::Program &program, const char *kernel_name);
    /**
     * @brief Creates an OpenCL buffer with the specified size and memory flags. If somehing goes wrong during buffer creation, a
     * detailed error message is printed to stderr and a runtime_error exception is thrown.
     * 
     * @param size The size of the buffer to create in bytes.
     * @param flags The memory flags for the buffer (e.g., CL_MEM_READ_WRITE).
     * @param host_ptr Optional pointer to host memory to initialize the buffer with (default is nullptr).
     * @return The created OpenCL buffer.
     */
    cl::Buffer createBuffer(size_t size, cl_mem_flags flags, void *host_ptr = nullptr);

    /**
     * @brief Executes the given OpenCL kernel with the specified global and local work sizes. This call is non-blocking.
     * 
     * @param kernel The OpenCL kernel to execute.
     * @param global_range The global work size as cl::NDRange.
     * @param local_range The local work size as cl::NDRange.
     */
    void executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range);

    /**
     * @brief Reads data from the given OpenCL buffer into the specified host memory. This call is blocking and 
     * will wait until the read operation is complete.
     * 
     * @param buffer The OpenCL buffer to read from.
     * @param host_ptr Pointer to the host memory where the data will be copied.
     * @param size The size of data to read in bytes.
     * @param offset The offset in the buffer from where to start reading (default is 0).
     */
    void readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, size_t offset = 0) const;
    /**
     * @brief Writes data from the specified host memory into the given OpenCL buffer. This call is blocking and
     * will wait until the write operation is complete.
     * 
     * @param buffer The OpenCL buffer to write to.
     * @param host_ptr Pointer to the host memory containing the data to write.
     * @param size The size of data to write in bytes.
     * @param offset The offset in the buffer from where to start writing (default is 0).
     */
    void writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, size_t offset = 0) const;
    /**
     * @brief Maps the given OpenCL buffer to pinned host memory, allowing for efficient data transfer between the host and device.
     * This call is blocking and will wait until the mapping operation is complete.
     * 
     * @param buffer The OpenCL buffer to map.
     * @param flags The mapping flags (e.g., CL_MAP_READ, CL_MAP_WRITE).
     * @param size The size of data to map in bytes.
     * @param offset The offset in the buffer from where to start mapping (default is 0).
     * @return A pointer to the mapped host memory.
     */
    void *mapHostPtrToPinnedMemory(const cl::Buffer &buffer, cl_map_flags flags, size_t size, size_t offset = 0) const;
    /**
     * @brief Unmaps the previously mapped OpenCL buffer from pinned host memory.
     * 
     * @param buffer The OpenCL buffer to unmap.
     * @param host_ptr Pointer to the mapped host memory to unmap.
     */
    void unMapHostPtr(const cl::Buffer &buffer, void *host_ptr) const;

    /**
     * @brief Synchronizes the OpenCL command queue, ensuring that all previously enqueued commands have completed.
     */
    void synchronize() const;

    // -- Getters --
    // I refuse to document these, they are self-explanatory and I'm lazy.

    std::string getBuildOptions() const { return build_options; }
    cl::Context getContext() const { return context; }
    cl::Device getSelectedDevice() const { return selected_device; }
    cl::Platform getSelectedPlatform() const { return selected_platform; }
    cl::CommandQueue getCommandQueue() const { return command_queue; }
};
