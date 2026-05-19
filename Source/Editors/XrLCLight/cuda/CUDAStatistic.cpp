#include "stdafx.h"
#include "CUDAContext.h"
#include <thread>
#include <mutex>

// GPU Usage
#include <nvml.h>
static bool isStatisticInitilized = false;
static nvmlDevice_t device = nullptr;

char* VVML_RESULT_INFO[] =
{
	"NVML_SUCCESS",
	"NVML_ERROR_UNINITIALIZED",
	"NVML_ERROR_INVALID_ARGUMENT",
	"NVML_ERROR_NOT_SUPPORTED",
	"NVML_ERROR_NO_PERMISSION",
	"NVML_ERROR_ALREADY_INITIALIZED",
	"NVML_ERROR_NOT_FOUND",
	"NVML_ERROR_INSUFFICIENT_SIZE",
	"NVML_ERROR_INSUFFICIENT_POWER",
	"NVML_ERROR_DRIVER_NOT_LOADED",
	"NVML_ERROR_TIMEOUT",
	"NVML_ERROR_IRQ_ISSUE",
	"NVML_ERROR_LIBRARY_NOT_FOUND",
	"NVML_ERROR_FUNCTION_NOT_FOUND",
	"NVML_ERROR_CORRUPTED_INFOROM",
	"NVML_ERROR_GPU_IS_LOST",
	"NVML_ERROR_RESET_REQUIRED",
	"NVML_ERROR_OPERATING_SYSTEM",
	"NVML_ERROR_LIB_RM_VERSION_MISMATCH",
	"NVML_ERROR_IN_USE",
	"NVML_ERROR_MEMORY",
	"NVML_ERROR_NO_DATA",
	"NVML_ERROR_VGPU_ECC_NOT_SUPPORTED",
	"NVML_ERROR_INSUFFICIENT_RESOURCES",
	"NVML_ERROR_FREQ_NOT_SUPPORTED",
	"NVML_ERROR_ARGUMENT_VERSION_MISMATCH",
	"NVML_ERROR_DEPRECATED",
	"NVML_ERROR_NOT_READY",
	"NVML_ERROR_GPU_NOT_FOUND",
	"NVML_ERROR_INVALID_STATE",
	"NVML_ERROR_RESET_TYPE_NOT_SUPPORTED"
};

void CudaUsage(unsigned int& UsageCuda, unsigned int& UsageMemory)
{
	nvmlReturn_t result;

	if (device != nullptr)
	{
		nvmlUtilization_t util{};
		result = nvmlDeviceGetUtilizationRates(device, &util);
		// Msg("result: %s", VVML_RESULT_INFO[result]);

		if (result == NVML_SUCCESS)
		{
			UsageCuda = util.gpu;
			UsageMemory = util.memory;
		}
 	}
}

static xr_vector<float> gpuUsage;
static xr_vector<float> memUsage;

xr_vector<float> get_cuda_usage()
{
	return gpuUsage;
}

xr_vector<float> get_mem_usage()
{
	return memUsage;
}

void CudaStatisticThread()
{
	nvmlReturn_t result = nvmlInit();
	if (!isStatisticInitilized)
	{
		if (result != NVML_SUCCESS)
		{
			Msg("NVML Init failed: ", nvmlErrorString(result));
			return;
		}

		unsigned int deviceCount = 0;
		nvmlDeviceGetCount(&deviceCount);

		if (deviceCount == 0)
		{
			Msg("No NVIDIA GPU found.");
			nvmlShutdown();
			return;
		}

		nvmlDeviceGetHandleByIndex(0, &device);

		char name[128];
		nvmlDeviceGetName(device, name, sizeof(name));
		Msg("--- Cuda Device: %s", name);
		isStatisticInitilized = true;
	}

	std::thread([] 
	{
		while (true)
		{
			u32 uCuda = 0;
			u32 uMemory = 0;
			CudaUsage(uCuda, uMemory);
			
			{
				gpuUsage.push_back(uCuda);

				if (gpuUsage.size() > 80)
				{
					gpuUsage.erase(gpuUsage.begin());
				}

				memUsage.push_back(uMemory);
 				if (memUsage.size() > 80)
				{
					memUsage.erase(memUsage.begin());
				}
			}

			Sleep(33);
 		};
	
	}).detach();
}

void CudaStatsShutdown()
{
	if (isStatisticInitilized)
		nvmlShutdown();
}

