#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
 
typedef unsigned int u32;
typedef unsigned char u8;

#define BORDER 1
const	unsigned int		alpha_ref = 254 - BORDER;

__global__ void checkAlphaKernel(
    const u8* surface_tbb, const u8 * lm,
    u32 SurfaceGrid,
    int s_x, int s_y,
    int Rax, int Ray,
    int* result_flag // 0 = true, 1 = false
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= s_x || y >= s_y)
        return;

    const unsigned char* P = surface_tbb + (y + Ray) * SurfaceGrid + Rax + x;
    const unsigned char* S = lm + y * s_x + x;

    if (*P && (*S >= alpha_ref)) 
    {
        atomicExch(result_flag, 1);
    }
}
 
extern "C" cudaError_t cuda_place(u32 SurfaceGrid, u32 RectX, u32 RectY, u32 s_x, u32 s_y, u8* surface, u8* lightmap, bool& isFineded)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((s_x + 15) / 16, (s_y + 15) / 16);

	// Выделение памяти
	int* d_result;
	cudaMalloc(&d_result, sizeof(int));
	cudaMemset(d_result, 0, sizeof(int)); // 0 = true (по умолчанию)
	cudaDeviceSynchronize();

	// Вызов ядра
	checkAlphaKernel <<< gridSize, blockSize >>> (
		surface, lightmap, SurfaceGrid,
		s_x, s_y, RectX, RectY, d_result
		);

	// Копирование результата обратно
	int h_result;
	cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);


	// Проверка результата
	bool is_valid = (h_result == 0);
	cudaFree(d_result);

	isFineded = is_valid;
	 
	return cudaDeviceSynchronize();
}