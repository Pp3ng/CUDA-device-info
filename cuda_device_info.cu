#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <iomanip>

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

// Format functions for memory size, frequency, and dimensions of threads and blocks for human-readable output
std::string formatMemorySize(size_t bytes)
{
       const char *units[] = {"B", "KB", "MB", "GB", "TB"};
       int unitIndex = 0;
       double size = static_cast<double>(bytes);

       while (size >= 1024 && unitIndex < 4)
       {
              size /= 1024;
              unitIndex++;
       }

       std::stringstream ss;
       ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
       return ss.str();
}

std::string formatFrequency(int freqInKHz)
{
       if (freqInKHz >= 1000000)
       {
              return std::to_string(freqInKHz / 1000000) + " GHz";
       }
       else if (freqInKHz >= 1000)
       {
              return std::to_string(freqInKHz / 1000) + " MHz";
       }
       return std::to_string(freqInKHz) + " kHz";
}

std::string formatDimension(int x, int y = 0, int z = 0)
{
       std::stringstream ss;
       if (z > 0)
       {
              ss << x << " x " << y << " x " << z;
       }
       else if (y > 0)
       {
              ss << x << " x " << y;
       }
       else
       {
              ss << x;
       }
       return ss.str();
}

int getCudaCoresPerSM(int major, int minor)
{
       switch (major)
       {
       case 2:
              return 32; // Fermi
       case 3:
              return 192; // Kepler
       case 5:
              return 128; // Maxwell
       case 6:
              return 64; // Pascal
       case 7:
              return 64; // Volta/Turing
       case 8:
              return 128; // Ampere
       case 9:
              return 128; // Hopper
       default:
              return 0;
       }
}
std::string getArchitectureName(int major, int minor)
{
       if (major == 2)
              return "Fermi";
       if (major == 3)
              return "Kepler";
       if (major == 5)
              return "Maxwell";
       if (major == 6)
              return "Pascal";
       if (major == 7)
       {
              if (minor == 0)
                     return "Volta";
              if (minor == 5)
                     return "Turing";
       }
       if (major == 8)
       {
              if (minor == 0)
                     return "Ampere";
              if (minor >= 6)
                     return "Ada Lovelace";
       }
       if (major == 9)
              return "Hopper";
       return "Unknown";
}
// Functions for printing headers, sub-headers, and key-value pairs
void printHeader(const char *header)
{
       printf("\n%s%s%s\n", BOLDCYAN, header, RESET);
}

void printSubHeader(const char *header)
{
       printf("%s%s%s\n", BOLDYELLOW, header, RESET);
}

void printValue(const char *label, const std::string &value)
{
       printf("%s%-45s%s%s%s\n", CYAN, label, RESET, value.c_str(), RESET);
}

void printValueBool(const char *label, bool value)
{
       printf("%s%-45s%s%s%s\n", CYAN, label, RESET, value ? "Yes" : "No", RESET);
}

// Format TFLOPS value as a string
std::string formatTFlops(float tflops)
{
       std::stringstream ss;
       ss << std::fixed << std::setprecision(2) << tflops << " TFLOPS";
       return ss.str();
}

// Format percentage value as a string
std::string formatPercentage(float value)
{
       std::stringstream ss;
       ss << std::fixed << std::setprecision(2) << value << "%";
       return ss.str();
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
              printValue("Compute Capability:", std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor) + " (" + getArchitectureName(deviceProp.major, deviceProp.minor) + ")");
              printValue("MultiProcessor Count:",
                         std::to_string(deviceProp.multiProcessorCount) + " SMs");
              printValue("Maximum Threads Per MultiProcessor:",
                         std::to_string(deviceProp.maxThreadsPerMultiProcessor) + " threads");

              printValue("CUDA Cores per SM:", std::to_string(getCudaCoresPerSM(deviceProp.major, deviceProp.minor)));
              printValue("Total CUDA Cores:", std::to_string(getCudaCoresPerSM(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount));
              printValue("Device Compute Mode:", deviceProp.computeMode == cudaComputeModeDefault            ? "Default"
                                                 : deviceProp.computeMode == cudaComputeModeExclusive        ? "Exclusive"
                                                 : deviceProp.computeMode == cudaComputeModeProhibited       ? "Prohibited"
                                                 : deviceProp.computeMode == cudaComputeModeExclusiveProcess ? "Exclusive Process"
                                                                                                             : "Unknown");

              printValueBool("Device Integrated:", deviceProp.integrated);
              printValueBool("Device TCC Driver:", deviceProp.tccDriver);
              printValue("CUDA Driver Version:", std::to_string([]
                                                                { int driverVersion; cudaDriverGetVersion(&driverVersion); return driverVersion; }()));
              printValue("Max Shared Memory Per Block:", formatMemorySize(deviceProp.sharedMemPerBlock));
              printValue("Max Grid Size:", formatDimension(deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]));
              printValue("Max Texture Gather Size:", std::to_string(deviceProp.maxTexture2DGather[0]) + " x " + std::to_string(deviceProp.maxTexture2DGather[1]));
              printValueBool("Unified Memory Support:", deviceProp.unifiedAddressing);

              // Memory Information
              printHeader("Memory Information:");
              printValue("Total Global Memory:", formatMemorySize(deviceProp.totalGlobalMem));
              printValue("Total Constant Memory:", formatMemorySize(deviceProp.totalConstMem));
              printValue("Shared Memory Per Block:", formatMemorySize(deviceProp.sharedMemPerBlock));
              printValue("Reserved Shared Memory Per Block:",
                         formatMemorySize(deviceProp.reservedSharedMemPerBlock));
              printValue("Memory Bus Width:", std::to_string(deviceProp.memoryBusWidth) + " bits");
              printValue("Memory Clock Rate:", formatFrequency(deviceProp.memoryClockRate));
              printValue("L2 Cache Size:", formatMemorySize(deviceProp.l2CacheSize));
              printValueBool("Global L1 Cache Supported:", deviceProp.globalL1CacheSupported);
              printValueBool("Local L1 Cache Supported:", deviceProp.localL1CacheSupported);
              printValueBool("Pageable Memory Access:", deviceProp.pageableMemoryAccess);
              printValueBool("Concurrent Managed Memory:", deviceProp.concurrentManagedAccess);

              // Advanced Memory Properties
              printHeader("Advanced Memory Properties:");
              printValue("Max Registers Per Block:", std::to_string(deviceProp.regsPerBlock));
#if CUDART_VERSION >= 12000
              printValueBool("Memory Pools Supported:", deviceProp.memoryPoolsSupported);
#endif
              printValue("Access Policy Max Window Size:",
                         formatMemorySize(deviceProp.accessPolicyMaxWindowSize));
              printValueBool("Host Register Supported:", deviceProp.hostRegisterSupported);
              printValueBool("Direct Managed Memory Access:",
                             deviceProp.directManagedMemAccessFromHost);

              // Thread and Block Information
              printHeader("Thread and Block Information:");
              printValue("Max Threads Per Block:",
                         std::to_string(deviceProp.maxThreadsPerBlock) + " threads");
              printValue("Max Thread Dimensions:",
                         formatDimension(deviceProp.maxThreadsDim[0],
                                         deviceProp.maxThreadsDim[1],
                                         deviceProp.maxThreadsDim[2]) +
                             " threads");
              printValue("Max Grid Dimensions:",
                         formatDimension(deviceProp.maxGridSize[0],
                                         deviceProp.maxGridSize[1],
                                         deviceProp.maxGridSize[2]) +
                             " blocks");
              printValue("Warp Size:", std::to_string(deviceProp.warpSize) + " threads");

              // Clock Information
              printHeader("Clock Information:");
              printValue("Clock Rate:", formatFrequency(deviceProp.clockRate));

              // Texture Information
              printHeader("Texture Memory Information:");
              printValue("Maximum 1D Texture Size:",
                         std::to_string(deviceProp.maxTexture1D) + " texels");
              printValue("Maximum 2D Texture Dimensions:",
                         formatDimension(deviceProp.maxTexture2D[0],
                                         deviceProp.maxTexture2D[1]) +
                             " texels");
              printValue("Maximum 3D Texture Dimensions:",
                         formatDimension(deviceProp.maxTexture3D[0],
                                         deviceProp.maxTexture3D[1],
                                         deviceProp.maxTexture3D[2]) +
                             " texels");
              printValue("Maximum 1D Layered Texture Size:",
                         formatDimension(deviceProp.maxTexture1DLayered[0],
                                         deviceProp.maxTexture1DLayered[1]) +
                             " texels");
              printValue("Maximum 2D Layered Texture Size:",
                         formatDimension(deviceProp.maxTexture2DLayered[0],
                                         deviceProp.maxTexture2DLayered[1],
                                         deviceProp.maxTexture2DLayered[2]) +
                             " texels");
              printValue("Texture Alignment:", formatMemorySize(deviceProp.textureAlignment));

              // Surface Information
              printHeader("Surface Memory Information:");
              printValue("Maximum 1D Surface Size:",
                         std::to_string(deviceProp.maxSurface1D) + " elements");
              printValue("Maximum 2D Surface Dimensions:",
                         formatDimension(deviceProp.maxSurface2D[0],
                                         deviceProp.maxSurface2D[1]) +
                             " elements");
              printValue("Maximum 3D Surface Dimensions:",
                         formatDimension(deviceProp.maxSurface3D[0],
                                         deviceProp.maxSurface3D[1],
                                         deviceProp.maxSurface3D[2]) +
                             " elements");
              printValue("Maximum 1D Layered Surface Size:",
                         formatDimension(deviceProp.maxSurface1DLayered[0],
                                         deviceProp.maxSurface1DLayered[1]) +
                             " elements");
              printValue("Maximum 2D Layered Surface Size:",
                         formatDimension(deviceProp.maxSurface2DLayered[0],
                                         deviceProp.maxSurface2DLayered[1],
                                         deviceProp.maxSurface2DLayered[2]) +
                             " elements");
              printValue("Surface Alignment:", formatMemorySize(deviceProp.surfaceAlignment));

              // Advanced Features
              printHeader("Advanced Features:");
              printValueBool("Concurrent Kernels:", deviceProp.concurrentKernels);
              printValueBool("Device Overlap Support:", deviceProp.deviceOverlap);
              printValue("Async Engine Count:", std::to_string(deviceProp.asyncEngineCount));
              printValueBool("Unified Addressing:", deviceProp.unifiedAddressing);
              printValueBool("Managed Memory:", deviceProp.managedMemory);
              printValueBool("Concurrent Managed Memory:", deviceProp.concurrentManagedAccess);
              printValueBool("Stream Priorities Supported:", deviceProp.streamPrioritiesSupported);
              printValueBool("Cooperative Launch:", deviceProp.cooperativeLaunch);
              printValueBool("Multi-Device Cooperative Launch:",
                             deviceProp.cooperativeMultiDeviceLaunch);

              // Kernel Execution Properties
              printHeader("Kernel Execution Properties:");
              printValueBool("Kernel Execution Timeout Enabled:",
                             deviceProp.kernelExecTimeoutEnabled);
#if CUDART_VERSION >= 11000
              printValue("Max Blocks Per MultiProcessor:",
                         std::to_string(deviceProp.maxBlocksPerMultiProcessor) + " blocks");
#endif

              // Hardware Features
              printHeader("Hardware Features:");
              printValueBool("ECC Enabled:", deviceProp.ECCEnabled);
              printValueBool("Is Multi-GPU Board:", deviceProp.isMultiGpuBoard);
              printValueBool("Can Map Host Memory:", deviceProp.canMapHostMemory);
              printValueBool("Can Use Host Pointer For Registered Memory:",
                             deviceProp.canUseHostPointerForRegisteredMem);
              if (deviceProp.major >= 7)
              {
                     printValueBool("Tensor Core Support:", true);
                     printValue("Tensor Core Generation:",
                                deviceProp.major == 7 ? "First Gen (Volta/Turing)" : deviceProp.major == 8 ? "Second Gen (Ampere)"
                                                                                 : deviceProp.major == 9   ? "Third Gen (Hopper)"
                                                                                                           : "Unknown");
              }
              else
              {
                     printValueBool("Tensor Core Support:", false);
              }

              // PCI Information
              printHeader("PCI Information:");
              printValue("PCI Bus ID:", std::to_string(deviceProp.pciBusID));
              printValue("PCI Device ID:", std::to_string(deviceProp.pciDeviceID));
              printValue("PCI Domain ID:", std::to_string(deviceProp.pciDomainID));

              // Persisting L2 Cache Properties
#if CUDART_VERSION >= 11000
              printHeader("Persisting L2 Cache Properties:");
              printValue("Persisting L2 Cache Max Size:",
                         formatMemorySize(deviceProp.persistingL2CacheMaxSize));
#endif

              // Performance Metrics
              printHeader("Performance Metrics:");

              // Calculate theoretical memory bandwidth
              printValue("Theoretical Memory Bandwidth:",
                         (std::stringstream() << std::fixed << std::setprecision(2)
                                              << (2.0f * (deviceProp.memoryClockRate * 1000.0f) *
                                                  (deviceProp.memoryBusWidth / 8.0f) / 1.0e9f) // Memory bandwidth = 2 * (memory clock rate) * (memory bus width / 8) / 1e9
                                              << " GB/s")
                             .str());

              // Calculate theoretical single-precision floating-point performance (TFLOPS)
              // Single-precision TFLOPS = 2 * (core clock rate) * (number of cormance (TFLOPS)
              printValue("Theoretical Single-Precision Performance:", formatTFlops((2.0f * deviceProp.clockRate * 1e-6f *
                                                                                    getCudaCoresPerSM(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount / 1000.0f)));

              // Calculate theoretical double-precision floating-point performance (TFLOPS)
              // Note: This is a rough estimate, actual ratio may vary by architecture
              // Double-precision TFLOPS = Single-precision TFLOPS / 2
              printValue("Theoretical Double-Precision Performance:", formatTFlops((2.0f * deviceProp.clockRate * 1e-6f *
                                                                                    getCudaCoresPerSM(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount / 1000.0f) /
                                                                                   2.0f));

              // Calculate SM utilization
              // SM utilization = (max threads per SM / max threads per block) * 100
              printValue("SM Utilization:", formatPercentage((static_cast<float>(deviceProp.maxThreadsPerMultiProcessor) / deviceProp.maxThreadsPerBlock) * 100.0f));

              printf("\n%s%s================================================%s\n",
                     BOLDMAGENTA, BOLD, RESET);
       }

       return EXIT_SUCCESS;
}