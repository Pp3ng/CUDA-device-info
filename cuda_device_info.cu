#include <cuda_runtime.h>
#include <iostream>

// Color codes for formatting
#define RESET "\033[0m"
#define BOLD "\033[1m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"

// Bold colors
#define BOLDRED "\033[1;31m"
#define BOLDGREEN "\033[1;32m"
#define BOLDYELLOW "\033[1;33m"
#define BOLDBLUE "\033[1;34m"
#define BOLDMAGENTA "\033[1;35m"
#define BOLDCYAN "\033[1;36m"
#define BOLDWHITE "\033[1;37m"

void printHeader(const char *header)
{
       printf("\n%s%s%s\n", BOLDCYAN, header, RESET);
}

void printSubHeader(const char *header)
{
       printf("%s%s%s\n", BOLDYELLOW, header, RESET);
}

void printValue(const char *label, const char *value)
{
       printf("%s%-40s%s%s%s\n", CYAN, label, RESET, value, RESET);
}

void printValueInt(const char *label, int value)
{
       printf("%s%-40s%s%d%s\n", CYAN, label, RESET, value, RESET);
}

void printValueLong(const char *label, size_t value)
{
       printf("%s%-40s%s%zu%s\n", CYAN, label, RESET, value, RESET);
}

void printValueFloat(const char *label, float value, int precision = 2)
{
       printf("%s%-40s%s%.*f%s\n", CYAN, label, RESET, precision, value, RESET);
}

void printValueBool(const char *label, bool value)
{
       printf("%s%-40s%s%s%s\n", CYAN, label, RESET, value ? "Yes" : "No", RESET);
}

int main(void)
{
       int deviceCount;
       cudaGetDeviceCount(&deviceCount);

       if (deviceCount == 0)
       {
              printf("%sNo CUDA devices found.%s\n", BOLDRED, RESET);
              return EXIT_FAILURE;
       }

       for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
       {
              cudaDeviceProp deviceProp;
              cudaGetDeviceProperties(&deviceProp, deviceIndex);

              // Device Header
              printf("\n%s%s========== CUDA Device #%d Information ==========%s\n",
                     BOLDMAGENTA, BOLD, deviceIndex, RESET);

              // Basic Device Information
              printHeader("Basic Device Information:");
              printValue("Device Name:", deviceProp.name);
              printf("%s%-40s%s%d.%d%s\n", CYAN, "Compute Capability:", RESET,
                     deviceProp.major, deviceProp.minor, RESET);
              printValueInt("MultiProcessor Count:", deviceProp.multiProcessorCount);
              printValueInt("Maximum Threads Per MultiProcessor:",
                            deviceProp.maxThreadsPerMultiProcessor);
              printValueInt("Device Compute Mode:", deviceProp.computeMode);
              printValueBool("Device Integrated:", deviceProp.integrated);
              printValueBool("Device TCC Driver:", deviceProp.tccDriver);

              // Memory Information
              printHeader("Memory Information:");
              printValueInt("Total Global Memory (MB):",
                            deviceProp.totalGlobalMem / (1024 * 1024));
              printValueLong("Total Constant Memory (bytes):", deviceProp.totalConstMem);
              printValueLong("Shared Memory Per Block (bytes):",
                             deviceProp.sharedMemPerBlock);
              printValueLong("Reserved Shared Memory Per Block (bytes):",
                             deviceProp.reservedSharedMemPerBlock);
              printValueInt("Memory Bus Width (bits):", deviceProp.memoryBusWidth);
              printValueInt("Memory Clock Rate (kHz):", deviceProp.memoryClockRate);
              printValueLong("L2 Cache Size (bytes):", deviceProp.l2CacheSize);
              printValueBool("Global L1 Cache Supported:",
                             deviceProp.globalL1CacheSupported);
              printValueBool("Local L1 Cache Supported:",
                             deviceProp.localL1CacheSupported);
              printValueBool("Pageable Memory Access:", deviceProp.pageableMemoryAccess);
              printValueBool("Concurrent Managed Memory:",
                             deviceProp.concurrentManagedAccess);

              // Advanced Memory Properties
              printHeader("Advanced Memory Properties:");
              printValueInt("Max Registers Per Block:", deviceProp.regsPerBlock);
#if CUDART_VERSION >= 12000
              printValueBool("Memory Pools Supported:", deviceProp.memoryPoolsSupported);
#endif
              printValueLong("Access Policy Max Window Size:",
                             deviceProp.accessPolicyMaxWindowSize);
              printValueBool("Host Register Supported:",
                             deviceProp.hostRegisterSupported);
              printValueBool("Pageable Memory Access:", deviceProp.pageableMemoryAccess);
              printValueBool("Direct Managed Memory Access:",
                             deviceProp.directManagedMemAccessFromHost);

              // Thread and Block Information
              printHeader("Thread and Block Information:");
              printValueInt("Max Threads Per Block:", deviceProp.maxThreadsPerBlock);
              printf("%s%-40s%s(%d, %d, %d)%s\n", CYAN, "Max Thread Dimensions:", RESET,
                     deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
                     deviceProp.maxThreadsDim[2], RESET);
              printf("%s%-40s%s(%d, %d, %d)%s\n", CYAN, "Max Grid Dimensions:", RESET,
                     deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
                     deviceProp.maxGridSize[2], RESET);
              printValueInt("Warp Size:", deviceProp.warpSize);

              // Clock Information
              printHeader("Clock Information:");
              printValueInt("Clock Rate (kHz):", deviceProp.clockRate);

              // Texture Information
              printHeader("Texture Memory Information:");
              printValueInt("Maximum 1D Texture Size:", deviceProp.maxTexture1D);
              printf("%s%-40s%s%d x %d%s\n", CYAN,
                     "Maximum 2D Texture Dimensions:", RESET, deviceProp.maxTexture2D[0],
                     deviceProp.maxTexture2D[1], RESET);
              printf("%s%-40s%s%d x %d x %d%s\n", CYAN,
                     "Maximum 3D Texture Dimensions:", RESET, deviceProp.maxTexture3D[0],
                     deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2], RESET);
              printf("%s%-40s%s%d x %d%s\n", CYAN,
                     "Maximum 1D Layered Texture Size:", RESET,
                     deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
                     RESET);
              printf("%s%-40s%s%d x %d x %d%s\n", CYAN,
                     "Maximum 2D Layered Texture Size:", RESET,
                     deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
                     deviceProp.maxTexture2DLayered[2], RESET);
              printValueInt("Texture Alignment (bytes):", deviceProp.textureAlignment);

              // Surface Information
              printHeader("Surface Memory Information:");
              printValueInt("Maximum 1D Surface Size:", deviceProp.maxSurface1D);
              printf("%s%-40s%s%d x %d%s\n", CYAN,
                     "Maximum 2D Surface Dimensions:", RESET, deviceProp.maxSurface2D[0],
                     deviceProp.maxSurface2D[1], RESET);
              printf("%s%-40s%s%d x %d x %d%s\n", CYAN,
                     "Maximum 3D Surface Dimensions:", RESET, deviceProp.maxSurface3D[0],
                     deviceProp.maxSurface3D[1], deviceProp.maxSurface3D[2], RESET);
              printf("%s%-40s%s%d x %d%s\n", CYAN,
                     "Maximum 1D Layered Surface Size:", RESET,
                     deviceProp.maxSurface1DLayered[0], deviceProp.maxSurface1DLayered[1],
                     RESET);
              printf("%s%-40s%s%d x %d x %d%s\n", CYAN,
                     "Maximum 2D Layered Surface Size:", RESET,
                     deviceProp.maxSurface2DLayered[0], deviceProp.maxSurface2DLayered[1],
                     deviceProp.maxSurface2DLayered[2], RESET);
              printValueInt("Surface Alignment (bytes):", deviceProp.surfaceAlignment);

              // Advanced Features
              printHeader("Advanced Features:");
              printValueBool("Concurrent Kernels:", deviceProp.concurrentKernels);
              printValueBool("Device Overlap Support:", deviceProp.deviceOverlap);
              printValueInt("Async Engine Count:", deviceProp.asyncEngineCount);
              printValueBool("Unified Addressing:", deviceProp.unifiedAddressing);
              printValueBool("Managed Memory:", deviceProp.managedMemory);
              printValueBool("Concurrent Managed Memory:",
                             deviceProp.concurrentManagedAccess);
              printValueBool("Stream Priorities Supported:",
                             deviceProp.streamPrioritiesSupported);
              printValueBool("Cooperative Launch:", deviceProp.cooperativeLaunch);
              printValueBool("Multi-Device Cooperative Launch:",
                             deviceProp.cooperativeMultiDeviceLaunch);

              // Kernel Execution Properties
              printHeader("Kernel Execution Properties:");
              printValueBool("Kernel Execution Timeout Enabled:",
                             deviceProp.kernelExecTimeoutEnabled);
#if CUDART_VERSION >= 11000
              printValueInt("Max Blocks Per MultiProcessor:",
                            deviceProp.maxBlocksPerMultiProcessor);
#endif

              // Hardware Features
              printHeader("Hardware Features:");
              printValueBool("ECC Enabled:", deviceProp.ECCEnabled);
              printValueBool("Is Multi-GPU Board:", deviceProp.isMultiGpuBoard);
              printValueBool("Can Map Host Memory:", deviceProp.canMapHostMemory);
              printValueBool("Can Use Host Pointer For Registered Memory:",
                             deviceProp.canUseHostPointerForRegisteredMem);

              // PCI Information
              printHeader("PCI Information:");
              printValueInt("PCI Bus ID:", deviceProp.pciBusID);
              printValueInt("PCI Device ID:", deviceProp.pciDeviceID);
              printValueInt("PCI Domain ID:", deviceProp.pciDomainID);

              // Persisting L2 Cache Properties (CUDA 11.0+)
#if CUDART_VERSION >= 11000
              printHeader("Persisting L2 Cache Properties:");
              printValueLong("Persisting L2 Cache Max Size (bytes):",
                             deviceProp.persistingL2CacheMaxSize);
#endif

              // Memory Bandwidth Calculation
              printHeader("Performance Metrics:");
              float memoryBandwidth = 2.0f * deviceProp.memoryClockRate *
                                      (deviceProp.memoryBusWidth / 8) / 1.0e6f;
              printValueFloat("Calculated Memory Bandwidth (GB/s):", memoryBandwidth);

              printf("\n%s%s================================================%s\n",
                     BOLDMAGENTA, BOLD, RESET);
       }

       return EXIT_SUCCESS;
}