#pragma once
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>


#define OPTIX_CHECK(call)                                             \
    do {                                                             \
        OptixResult res = call;                                       \
        if (res != OPTIX_SUCCESS) {                                   \
            Msg("! Optix call '%s' failed with code %d\n",  \
                    #call, (int)res);                                 \
        }                                                            \
    } while (0)
#define CUDA_CHECK(call)                                                      \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            Msg("! CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_2(call)                                                       \
    do {                                                                         \
        CUresult err = (call);                                                   \
        if (err != CUDA_SUCCESS) {                                               \
            const char* errStr = nullptr;                                        \
            cuGetErrorString(err, &errStr);                                      \
            Msg("! CUDA Error at %s:%d: %s (code %d)\n",                         \
                __FILE__, __LINE__,                                              \
                (errStr ? errStr : "Unknown CUDA error"),                        \
                err);                                                             \
        }                                                                        \
    } while (0)
 
class OptixContext
{
private: 
    CUdevice cuDev;
    CUcontext cudaContext = nullptr;
    OptixDeviceContext optixContext = nullptr;
    OptixPipeline m_pipeline = nullptr;
    OptixShaderBindingTable m_sbt = {};
    int cudaDeviceId = 0;
 
public:
    bool Initialize();
    void Destroy();

    void CreatePipeline(const char* ptxCode);

    OptixDeviceContext GetOptixContext() const { return optixContext; }
 
    static void OptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
    {
        string4096 formattedMsg;
        sprintf_s(formattedMsg, "* [OptiX][%s]: %s", tag, message);

        R_ASSERT(xr_strlen(message) < 4096);

        switch (level) {
        case 1: Log(formattedMsg); break;    // FATAL
        case 2: Log(formattedMsg); break;    // ERROR
        case 3: Log(formattedMsg); break;    // WARNING
        case 4: Log(formattedMsg); break;    // INFO
        default: break;
        }     
    }

    OptixPipeline GetPipeline() const { return m_pipeline; }
    OptixShaderBindingTable& GetSBT() { return m_sbt; }

    // Создание CUDA stream
    static CUstream CreateCudaStream()
    {
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        return stream;
    }

    // Уничтожение CUDA stream
    static void DestroyCudaStream(CUstream stream)
    {
        if (stream) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }
};
