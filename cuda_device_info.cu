#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <utility>

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

// Architecture information structure
struct ArchInfo {
    std::string name;
    int cudaCoresPerSM;
    std::string generation;
    
    ArchInfo() : name("Unknown"), cudaCoresPerSM(0), generation("Unknown") {}
    ArchInfo(const std::string& n, int cores, const std::string& gen) 
        : name(n), cudaCoresPerSM(cores), generation(gen) {}
};

// Global architecture information map
std::map<std::pair<int, int>, ArchInfo> architectureMap = {
    // Fermi (2.x)
    {{2, 0}, ArchInfo("Fermi", 32, "First Generation")},
    {{2, 1}, ArchInfo("Fermi", 32, "First Generation")},
    
    // Kepler (3.x)
    {{3, 0}, ArchInfo("Kepler", 192, "Second Generation")},
    {{3, 2}, ArchInfo("Kepler", 192, "Second Generation")},
    {{3, 5}, ArchInfo("Kepler", 192, "Second Generation")},
    {{3, 7}, ArchInfo("Kepler", 192, "Second Generation")},
    
    // Maxwell (5.x)
    {{5, 0}, ArchInfo("Maxwell", 128, "Third Generation")},
    {{5, 2}, ArchInfo("Maxwell", 128, "Third Generation")},
    {{5, 3}, ArchInfo("Maxwell", 128, "Third Generation")},
    
    // Pascal (6.x)
    {{6, 0}, ArchInfo("Pascal", 64, "Fourth Generation")},
    {{6, 1}, ArchInfo("Pascal", 64, "Fourth Generation")},
    {{6, 2}, ArchInfo("Pascal", 64, "Fourth Generation")},
    
    // Volta (7.0)
    {{7, 0}, ArchInfo("Volta", 64, "Fifth Generation")},
    
    // Turing (7.5)
    {{7, 5}, ArchInfo("Turing", 64, "Fifth Generation")},
    
    // Ampere (8.x)
    {{8, 0}, ArchInfo("Ampere", 128, "Sixth Generation")},
    {{8, 6}, ArchInfo("Ada Lovelace", 128, "Sixth Generation")},
    {{8, 7}, ArchInfo("Ada Lovelace", 128, "Sixth Generation")},
    {{8, 9}, ArchInfo("Ada Lovelace", 128, "Sixth Generation")},
    
    // Hopper (9.x)
    {{9, 0}, ArchInfo("Hopper", 128, "Seventh Generation")}
};

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

// Get architecture information from the map
ArchInfo getArchitectureInfo(int major, int minor)
{
    auto key = std::make_pair(major, minor);
    auto it = architectureMap.find(key);
    
    if (it != architectureMap.end()) {
        return it->second;
    }
    
    // Fallback: try to find by major version only
    for (const auto& arch : architectureMap) {
        if (arch.first.first == major) {
            return arch.second;
        }
    }
    
    // Return unknown architecture info
    return ArchInfo("Unknown", 0, "Unknown");
}

int getCudaCoresPerSM(int major, int minor)
{
    return getArchitectureInfo(major, minor).cudaCoresPerSM;
}

std::string getArchitectureName(int major, int minor)
{
    return getArchitectureInfo(major, minor).name;
}
// Functions for printing headers, sub-headers, and key-value pairs

void printHeader(const char *header)
{
       printf("\n%s%s%s\n", BOLDYELLOW, header, RESET);
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

// Get architecture-specific double-precision ratio
float getDoublePrecisionRatio(int major, int minor)
{
    ArchInfo info = getArchitectureInfo(major, minor);
    
    // Architecture-specific FP64/FP32 ratios
    if (info.name == "Fermi") return 0.5f;      // 1:2 ratio
    if (info.name == "Kepler") return 0.33f;    // 1:3 ratio  
    if (info.name == "Maxwell") return 0.03f;   // 1:32 ratio
    if (info.name == "Pascal") return 0.03f;    // 1:32 ratio for consumer, 1:2 for Tesla
    if (info.name == "Volta") return 0.5f;      // 1:2 ratio for Tesla
    if (info.name == "Turing") return 0.03f;    // 1:32 ratio
    if (info.name == "Ampere") return 0.5f;     // 1:2 ratio for A100, 1:32 for others
    if (info.name == "Ada Lovelace") return 0.03f; // 1:32 ratio
    if (info.name == "Hopper") return 0.5f;     // 1:2 ratio
    
    return 0.5f; // Default fallback
}

// Calculate theoretical memory bandwidth in GB/s 
float calculateMemoryBandwidth(const cudaDeviceProp& prop)
{
    return 2.0f * (prop.memoryClockRate * 1000.0f) * (prop.memoryBusWidth / 8.0f) / 1.0e9f;
}

// Calculate theoretical compute performance
struct PerformanceMetrics {
    float singlePrecisionTFLOPS;
    float doublePrecisionTFLOPS;
    float halfPrecisionTFLOPS;
    float integerTOPS;
    float memoryBandwidth;
    float computeToMemoryRatio;
};

PerformanceMetrics calculatePerformanceMetrics(const cudaDeviceProp& prop, const ArchInfo& archInfo)
{
    PerformanceMetrics metrics;
    
    int cudaCores = archInfo.cudaCoresPerSM * prop.multiProcessorCount;
    float baseClockGHz = prop.clockRate * 1e-6f;
    
    // Single precision: 1 FMA = 2 operations per core per clock
    metrics.singlePrecisionTFLOPS = 2.0f * baseClockGHz * cudaCores / 1000.0f;
    
    // Double precision: architecture dependent
    float dpRatio = getDoublePrecisionRatio(prop.major, prop.minor);
    metrics.doublePrecisionTFLOPS = metrics.singlePrecisionTFLOPS * dpRatio;
    
    // Half precision: 2x single precision (assuming Tensor cores not counted)
    metrics.halfPrecisionTFLOPS = metrics.singlePrecisionTFLOPS * 2.0f;
    
    // Integer operations: same as single precision
    metrics.integerTOPS = metrics.singlePrecisionTFLOPS;
    
    // Memory bandwidth
    metrics.memoryBandwidth = calculateMemoryBandwidth(prop);
    
    // Compute to memory ratio (operations per byte)
    metrics.computeToMemoryRatio = (metrics.singlePrecisionTFLOPS * 1000.0f) / metrics.memoryBandwidth;
    
    return metrics;
}

int main(void)
{
       int deviceCount;
       cudaError_t error = cudaGetDeviceCount(&deviceCount);
       
       if (error != cudaSuccess)
       {
              printf("%sError getting device count: %s%s\n", BOLDRED, cudaGetErrorString(error), RESET);
              return EXIT_FAILURE;
       }

       if (deviceCount == 0)
       {
              printf("%sNo CUDA devices found.%s\n", BOLDRED, RESET);
              return EXIT_FAILURE;
       }
       
       // Print CUDA runtime version
       int runtimeVersion;
       cudaRuntimeGetVersion(&runtimeVersion);
       printf("%s%sCUDA Runtime Version: %d.%d%s\n", BOLDGREEN, BOLD, 
              runtimeVersion / 1000, (runtimeVersion % 100) / 10, RESET);
              
       int driverVersion;
       cudaDriverGetVersion(&driverVersion);
       printf("%s%sCUDA Driver Version: %d.%d%s\n", BOLDGREEN, BOLD,
              driverVersion / 1000, (driverVersion % 100) / 10, RESET);

       for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex)
       {
              cudaDeviceProp deviceProp;
              cudaGetDeviceProperties(&deviceProp, deviceIndex);

              // Device Header
              printf("\n%s%s========== CUDA Device #%d Information ==========%s\n",
                     BOLDMAGENTA, BOLD, deviceIndex, RESET);              // Basic Device Information
              printHeader("Basic Device Information:");
              printValue("Device Name:", deviceProp.name);
              
              ArchInfo archInfo = getArchitectureInfo(deviceProp.major, deviceProp.minor);
              printValue("Compute Capability:", std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor) + " (" + archInfo.name + ")");
              printValue("Architecture Generation:", archInfo.generation);
              printValue("MultiProcessor Count:",
                         std::to_string(deviceProp.multiProcessorCount) + " SMs");
              printValue("Maximum Threads Per MultiProcessor:",
                         std::to_string(deviceProp.maxThreadsPerMultiProcessor) + " threads");

              printValue("CUDA Cores per SM:", std::to_string(archInfo.cudaCoresPerSM));
              printValue("Total CUDA Cores:", std::to_string(archInfo.cudaCoresPerSM * deviceProp.multiProcessorCount));
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
#endif              // Performance Metrics
              printHeader("Performance Metrics:");

              // Calculate performance metrics using the new function
              PerformanceMetrics metrics = calculatePerformanceMetrics(deviceProp, archInfo);
              
              printValue("Theoretical Memory Bandwidth:", 
                         (std::stringstream() << std::fixed << std::setprecision(2) 
                                              << metrics.memoryBandwidth << " GB/s").str());

              printValue("Theoretical Single-Precision Performance:", formatTFlops(metrics.singlePrecisionTFLOPS));
              printValue("Theoretical Double-Precision Performance:", formatTFlops(metrics.doublePrecisionTFLOPS));
              printValue("Theoretical Half-Precision Performance:", formatTFlops(metrics.halfPrecisionTFLOPS));
              printValue("Theoretical Integer Operations Performance:", formatTFlops(metrics.integerTOPS) + " TOPS");
              
              // Additional performance metrics
              printValue("Compute to Memory Ratio:", 
                         (std::stringstream() << std::fixed << std::setprecision(2) 
                                              << metrics.computeToMemoryRatio << " ops/byte").str());
              
              // Calculate SM utilization percentage
              float smUtilization = (static_cast<float>(deviceProp.maxThreadsPerMultiProcessor) / deviceProp.maxThreadsPerBlock) * 100.0f;
              printValue("Max SM Utilization:", formatPercentage(smUtilization));
              
              // Calculate memory efficiency metrics
              float totalGlobalMemGB = deviceProp.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
              float memoryPerSM = totalGlobalMemGB / deviceProp.multiProcessorCount;
              printValue("Memory per SM:", 
                         (std::stringstream() << std::fixed << std::setprecision(2) 
                                              << memoryPerSM << " GB").str());
              
              // Theoretical occupancy metrics
              int maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
              printValue("Max Blocks per SM:", std::to_string(maxBlocksPerSM) + ((maxBlocksPerSM > 1) ? " blocks" : " block"));
              
              // Warp efficiency
              int warpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
              printValue("Max Warps per SM:", std::to_string(warpsPerSM) + " warps");

              // Architecture-Specific Information
              printHeader("Architecture-Specific Information:");
              if (deviceProp.major >= 7)
              {
                     printValue("Independent Thread Scheduling:", "Supported (Compute Capability 7.0+)");
              }
              else
              {
                     printValue("Independent Thread Scheduling:", "Not supported (Compute Capability < 7.0)");
              }
              if (deviceProp.major >= 8)
              {
                     printValue("Sparse CUDA Array Support:", deviceProp.sparseCudaArraySupported ? "Yes" : "No");
              }
              if (deviceProp.major >= 9)
              {
                     printValue("Dynamic Shared Memory Per Block:", formatMemorySize(deviceProp.sharedMemPerBlockOptin));
              }

              // Power and Thermal Information
              printHeader("Power and Thermal Information:");
              printValue("Power Management:", deviceProp.managedMemory ? "Supported" : "Not Supported");

              printf("\n%s%s================================================%s\n",
                     BOLDMAGENTA, BOLD, RESET);
       }

       return EXIT_SUCCESS;
}