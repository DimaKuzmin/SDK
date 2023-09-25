////////////////////////////////////
// Author: se7kills
// Date:   September 2023
// Proj:   SDK By RedPandas
// Desc:   Hardware light support class
 
#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif

#include "cuda_runtime.h" 

#include "../../xrCore/_types.h"
#include "xrRayDefinition.h"

 
// NO TEXTURES

/*
__global__ void ProcessHits(xrHardwareLCGlobalData* GlobalData, Ray* RayBuffer, Hit* HitBuffer,
	float* ColorBuffer, char* RayStatusBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, bool IsFirstTime,
	bool CheckRGB, bool CheckSun, bool CheckHemi, bool SkipFaceMode, u64* FacesToSkip, u32 * alive_rays_count, int Rounds)
*/

__device__ void ShiftRay(Ray* CurrentRay, Hit* InHit)
{
	//there is a not only shift, but strech tmax
	float& Distance = InHit->Distance;

	if (CurrentRay->tmax > Distance)
	{
		CurrentRay->tmax -= Distance;
	}
	else
	{
		CurrentRay->tmax = 0.0f;
	}

	CurrentRay->Origin.Mad_Self(CurrentRay->Direction, Distance + 0.1f);
}


__global__ void ProcessHits_NO_TEXTURE(Ray* RayBuffer, Hit* HitBuffer, int* alive_rays, size_t size_hits)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
 
	if (idx >= size_hits)
		return;	 
 
	Ray* OurRay = &RayBuffer[idx];
	Hit* OurHit = &HitBuffer[idx];

	if (OurHit->triId == -1)
	{
		 OurRay->tmax = 0;
	}
	else 
	{
		ShiftRay(OurRay, OurHit);
		atomicAdd(&alive_rays[0], 1);
	}
}


// Error Ignore
extern "C" cudaError_t RunProcessHits_NO_TEXTURE(Ray* RayBuffer, Hit* HitBuffer, int* alive_rays, size_t size_hits)
{
	int BlockSize = 1024;
	int GridSize = (size_hits / BlockSize) + 1;

	ProcessHits_NO_TEXTURE << <GridSize, BlockSize >> > 
	(
		RayBuffer, HitBuffer, alive_rays, size_hits 
	);

	return cudaDeviceSynchronize();
}


