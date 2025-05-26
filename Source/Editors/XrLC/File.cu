
#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif

#include "cuda_runtime.h"


#define BORDER 1
const	unsigned int		alpha_ref = 254 - BORDER;
typedef unsigned int u32;
typedef unsigned char u8;


#include <windows.h>
#include <stdio.h>
#include <stdarg.h>

#define LOG_SIZE 64*1024
#define LOG_STRLEN 64
__device__ char log_buffer[LOG_SIZE][LOG_STRLEN];

__global__ void checkAlphaKernel(
	const u8* surface_tbb, const u8* lightmap,
	u32 SurfaceGrid,
	u32 SizeX, u32 SizeY,
	u32 RectX, u32 RectY,
	int* result_flag // 0 = true, 1 = false
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = SizeX * SizeY;

	if (idx >= total) return;

	u32 x = idx % SizeX;
	u32 y = idx / SizeX;

	const u8* P = surface_tbb + (y + RectY) * SurfaceGrid + RectX + x;
	const u8* S = lightmap + y * SizeX + x;

	if (*P && (*S >= alpha_ref))
	{
		atomicExch(result_flag, 1);
	}

  	snprintf(log_buffer[idx], 128, "TID:%d X: %u, Y: %u", idx, x, y);
}

extern "C" cudaError_t cuda_place(u32 SurfaceGrid, u32 RectX, u32 RectY, u32 SizeX, u32 SizeY, u8* surface, u8* lightmap, bool& isFineded)
{
	dim3 blockSize(1024);
	dim3 gridSize((SizeX * SizeY + blockSize.x - 1) / blockSize.x);

	// Выделение памяти
	// 0 = true (по умолчанию)

	int* cuda_result;
	cudaMalloc(&cuda_result, sizeof(int));
	cudaMemset(cuda_result, 0, sizeof(int));

	// Вызов ядра
	checkAlphaKernel << < gridSize, blockSize >> >
		(
			surface, lightmap,
			SurfaceGrid,
			SizeX, SizeY,
			RectX, RectY,
			cuda_result
			);

	// Копирование результата обратно
	int h_result;
	cudaMemcpy(&h_result, cuda_result, sizeof(int), cudaMemcpyDeviceToHost);


	// Проверка результата
	bool is_valid = (h_result == 0);
	cudaFree(cuda_result);

	char host_log[LOG_SIZE][LOG_STRLEN];
	cudaMemcpyFromSymbol(&host_log, log_buffer, sizeof(host_log));

	for (int i = 0; i < LOG_SIZE; ++i)
	{
		if (host_log[i][0]) // если не пусто
			OutputDebugStringA(host_log[i]);
	}

	isFineded = is_valid;
	return cudaDeviceSynchronize();
}