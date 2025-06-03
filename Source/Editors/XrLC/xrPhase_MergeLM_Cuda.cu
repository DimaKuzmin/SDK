
#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif
 
/*
#include "cuda_runtime.h"

#define BORDER 1
const	unsigned int		alpha_ref = 254 - BORDER;
typedef unsigned int u32;
typedef unsigned char u8;


#include <windows.h>
#include <stdio.h>
#include <stdarg.h>

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
		atomicExch(result_flag, 1);
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
	  
	isFineded = is_valid;
	return cudaDeviceSynchronize();
}
*/

// cuda_rect_placement.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BORDER 1
#define SurfaceGrid 4096
 
#define NUM_RECTS 4096

typedef unsigned char uint8_t;

struct L_rect {
    int a_x, a_y, b_x, b_y;
    __host__ __device__ void init(int x1, int y1, int x2, int y2) {
        a_x = x1; a_y = y1; b_x = x2; b_y = y2;
    }
};

__device__ bool can_place_rect(
    const uint8_t* lm,
    const uint8_t* surface,
    int s_x, int s_y,
    int surf_grid, int alpha_ref,
    int R_x, int R_y)
{
    for (int y = 0; y < s_y; y++) {
        const uint8_t* S = lm + y * s_x;
        const uint8_t* P = surface + (y + R_y) * surf_grid + R_x;
        for (int x = 0; x < s_x; x++) {
            if (P[x] && S[x] >= alpha_ref)
                return false;
        }
    }
    return true;
}

__device__ void register_rect(
    uint8_t* surface,
     const uint8_t* lm,
    int s_x, int s_y,
    int surf_grid, int alpha_ref,
    int R_x, int R_y)
{
    for (int y = 0; y < s_y; y++) {
        uint8_t* P = surface + (y + R_y) * surf_grid + R_x;
        const uint8_t* S = lm + y * s_x;
        for (int x = 0; x < s_x; x++) {
            if (S[x] >= alpha_ref) 
            {
                P[x] = 255;
            }
        }
    }
}

__global__ void try_place_rects(
    L_rect* rects_out,
    uint8_t* surface,
    const uint8_t* lm,
    int lm_width, int lm_height,
    int surf_grid,
    int alpha_ref,
    bool* placed_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_RECTS) return;

    int SizeX = lm_width + 2 * BORDER;
    int SizeY = lm_height + 2 * BORDER;
    int x_max = surf_grid - SizeX;
    int y_max = surf_grid - SizeY;
    int y_max_line = surf_grid * 0.9;

    for (int y = 0; y < y_max; y++) 
    {
        for (int x = 0; x < x_max; x++) 
        {
            if (can_place_rect(lm, surface, SizeX, SizeY, surf_grid, alpha_ref, x, y))
            {
                register_rect(surface, lm, SizeX, SizeY, surf_grid, alpha_ref, x, y);
                rects_out[idx].init(x, y, x + SizeX, y + SizeY);
                placed_flags[idx] = true;
                return;
            }
        }
    }
    placed_flags[idx] = false;
}

int main() 
{
    const int lm_w = 32, lm_h = 32; // example size
    const int lm_total = (lm_w + 2 * BORDER) * (lm_h + 2 * BORDER);

    // Allocate host and device memory
    uint8_t* d_surface, * d_lm;
     L_rect* d_rects;
    bool* d_flags;

    cudaMalloc(&d_surface, SurfaceGrid * SurfaceGrid);
    cudaMemset(d_surface, 0, SurfaceGrid * SurfaceGrid);
  
    cudaMalloc(&d_lm, lm_total);
    cudaMemset(d_lm, 1, lm_total); // fill with alpha=1 (just for test)

    cudaMalloc(&d_rects, sizeof(L_rect) * NUM_RECTS);
    cudaMalloc(&d_flags, sizeof(bool) * NUM_RECTS);

    dim3 threads(1024);
    dim3 blocks((NUM_RECTS + threads.x - 1) / threads.x);
    try_place_rects << <blocks, threads >> > (d_rects, d_surface, d_occupied_y, d_lm, lm_w, lm_h, SurfaceGrid, alpha_ref, d_flags);

    // Copy result
    L_rect h_rects[NUM_RECTS];
    bool h_flags[NUM_RECTS];
    cudaMemcpy(h_rects, d_rects, sizeof(L_rect) * NUM_RECTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flags, d_flags, sizeof(bool) * NUM_RECTS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_RECTS; ++i) {
        if (h_flags[i]) {
            printf("Rect %d placed at: (%d,%d) to (%d,%d)\n", i, h_rects[i].a_x, h_rects[i].a_y, h_rects[i].b_x, h_rects[i].b_y);
        }
        else {
            printf("Rect %d could not be placed.\n", i);
        }
    }

    // Cleanup
    cudaFree(d_surface); 
    cudaFree(d_lm);
    cudaFree(d_rects);
    cudaFree(d_flags);

    return 0;
}
