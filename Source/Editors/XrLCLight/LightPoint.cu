////////////////////////////////////
// Author: Giperion
// Date:   September - October 2016
// Proj:   xrLC Redux project
// Desc:   Hardware light support class
// Modifer: ForserX

#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "../../xrCore/_types.h"
#include "xrRayDefinition.h"
#include "light_point.h"
#include "base_color.h"

__device__ void GetLightTypeAndIndex(xrHardwareLCGlobalData* GlobalData, u32 CurrentRayID,
	bool CheckRGB, bool CheckSun, bool CheckHemi,
	LightSource& LightType, int& LinearLightIndex)
{
	int LightID = -1;

	//fast routines
	if (!CheckRGB && !CheckSun)
	{
		LightType = LS_HEMI;
		LightID = CurrentRayID;
	}
	if (!CheckRGB && !CheckHemi)
	{
		LightType = LS_SUN;
		LightID = CurrentRayID;
	}
	if (!CheckSun && !CheckHemi)
	{
		LightType = LS_RGB;
		LightID = CurrentRayID;
	}

	//complex... shit
	///#REFACTOR: I know that somewhere a good algorithm to handle this situation
	if (LightType == LS_UNKNOWN)
	{
		if (CheckRGB)
		{
			if (CurrentRayID < GlobalData->LightSize->RGBLightCount)
			{
				LightType = LS_RGB;
				LightID = CurrentRayID;
				goto GotLightType;
			}
		}

		if (CheckSun)
		{
			if (CurrentRayID < (GlobalData->LightSize->SunLightCount + GlobalData->LightSize->RGBLightCount))
			{
				LightType = LS_SUN;
				LightID = CheckRGB ? CurrentRayID - GlobalData->LightSize->RGBLightCount : CurrentRayID;
				goto GotLightType;
			}
		}

		if (CheckHemi)
		{
			LightType = LS_HEMI;

			if (CheckRGB)
			{
				if (CheckSun)
				{
					LightID = CurrentRayID - (GlobalData->LightSize->SunLightCount + GlobalData->LightSize->RGBLightCount);
				}
				else
				{
					LightID = CurrentRayID - GlobalData->LightSize->RGBLightCount;
				}
			}
			else if (CheckSun)
			{
				LightID = CurrentRayID - GlobalData->LightSize->SunLightCount;
			}
			else
			{
				LightID = CurrentRayID;
			}
		}

	}

	//Get linear light index
GotLightType:
	if (LightType == LS_RGB)
	{
		LinearLightIndex = LightID;
	}
	else if (LightType == LS_SUN)
	{
		LinearLightIndex = GlobalData->LightSize->RGBLightCount + LightID;
	}
	else if (LightType == LS_HEMI)
	{
		LinearLightIndex = GlobalData->LightSize->RGBLightCount + GlobalData->LightSize->SunLightCount + LightID;
	}
}

__global__ void GenerateRaysForTask(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, Ray* RayBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, bool CheckRGB, bool CheckSun, bool CheckHemi)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// early exit condition
	if (idx >= AliveRaysCount) return;

	//check what current vertex id....
	int RaysPerVertex = 0;
	if (CheckRGB)  RaysPerVertex += GlobalData->LightSize->RGBLightCount;
	if (CheckSun)  RaysPerVertex += GlobalData->LightSize->SunLightCount;
	if (CheckHemi) RaysPerVertex += GlobalData->LightSize->HemiLightCount;

	//decode ray index in to vertex | lightType | lightLinearIndex
	u32 AliveRayIndex = AliveRaysIndexes[idx];
	u32 CurrentSurfaceID = AliveRayIndex / RaysPerVertex;
	u32 CurrentRayForVertex = AliveRayIndex % RaysPerVertex;

	LightSource LightType = LS_UNKNOWN;
	int LinearLightIndex = 0;
	GetLightTypeAndIndex(GlobalData, CurrentRayForVertex, CheckRGB, CheckSun, CheckHemi, LightType, LinearLightIndex);

	R_Light& CurrentLight = GlobalData->LightData[LinearLightIndex];

	HardwareVector CurrentSurfacePosition = RequestedRays[CurrentSurfaceID].Position;
	HardwareVector CurrentSurfaceNormal = RequestedRays[CurrentSurfaceID].Normal;

	//GSC brought me to here
	HardwareVector SurfacePositionMoved = CurrentSurfacePosition;
	SurfacePositionMoved.Mad_Self(CurrentSurfaceNormal, 0.01f);

	Ray& TargetRay = RayBuffer[idx];

	switch (CurrentLight.type)
	{
	case LT_DIRECT:
	{
		HardwareVector LightDir = CurrentLight.direction;
		LightDir = LightDir.Inverted();

		TargetRay.Origin = SurfacePositionMoved;
		TargetRay.Direction = LightDir;
		TargetRay.tmin = 0.0f;
		TargetRay.tmax = 1000.0f;
	}
	break;

	case LT_POINT:
	{
		HardwareVector LightDir = CurrentLight.position.Subtract(CurrentSurfacePosition);
		LightDir.Normalize_Safe();

		float DistanceSquared = CurrentSurfacePosition.DistanceSquared(CurrentLight.position);
		float Distance = sqrtf(DistanceSquared);

		TargetRay.Origin = SurfacePositionMoved;
		TargetRay.Direction = LightDir;
		TargetRay.tmin = 0.0f;
		TargetRay.tmax = Distance;
	}
	break;

	case LT_SECONDARY:
	{
		HardwareVector LightDir = CurrentLight.position.Subtract(CurrentSurfacePosition);
		LightDir.Normalize_Safe();

		float DistanceSquared = CurrentSurfacePosition.DistanceSquared(CurrentLight.position);
		float Distance = sqrtf(DistanceSquared);

		TargetRay.Origin = SurfacePositionMoved;
		TargetRay.Direction = LightDir;
		TargetRay.tmin = 0.0f;
		TargetRay.tmax = Distance;
	}
	break;
	}
}

//Ray Intersection

__device__ bool GetRayOptimizeOut(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, u32 SurfaceID, int LightID)
{
	HardwareVector& CurrentVertex = RequestedRays[SurfaceID].Position;
	HardwareVector& CurrentVertexNormal = RequestedRays[SurfaceID].Normal;
	R_Light& CurrentLight = GlobalData->LightData[LightID];

	if (CurrentLight.type == LT_DIRECT)
	{
		HardwareVector InvertedLightDir = CurrentLight.direction.Inverted();

		float LightInfluenceWeight = InvertedLightDir.DotProduct(CurrentVertexNormal);
		if (LightInfluenceWeight <= 0) return true;

	}
	else if (CurrentLight.type == LT_POINT)
	{
		float DistanceSquared = CurrentVertex.DistanceSquared(CurrentLight.position);
		if (DistanceSquared > CurrentLight.range2) return true;

		HardwareVector ProcessedLightDir = CurrentLight.position.Subtract(CurrentVertex);
		ProcessedLightDir.Normalize_Safe();
		float LightInfluenceWeight = ProcessedLightDir.DotProduct(CurrentVertexNormal);
		if (LightInfluenceWeight <= 0.0f) return true;
	}
	else if (CurrentLight.type == LT_SECONDARY)
	{
		float DistanceSquared = CurrentVertex.DistanceSquared(CurrentLight.position);
		if (DistanceSquared > CurrentLight.range2) return true;

		HardwareVector ProcessedLightDir = CurrentLight.position.Subtract(CurrentVertex);
		ProcessedLightDir.Normalize_Safe();
		float LightInfluenceWeight = ProcessedLightDir.DotProduct(CurrentVertexNormal);
		if (LightInfluenceWeight <= 0.0f) return true;

		LightInfluenceWeight *= -ProcessedLightDir.DotProduct(CurrentLight.direction);
		if (LightInfluenceWeight <= 0.0f) return true;
	}

	return false;
}

__global__ void CheckRayOptimizeOut(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, char* StatusBuffer, u64 MaxPossibleRays, bool CheckRGB, bool CheckSun, bool CheckHemi)
{
	u64 idx = threadIdx.x + blockIdx.x * blockDim.x;

	//early exit condition
	if (idx >= MaxPossibleRays) return;

	//check what current vertex id....
	int RaysPerVertex = 0;
	if (CheckRGB)  RaysPerVertex += GlobalData->LightSize->RGBLightCount;
	if (CheckSun)  RaysPerVertex += GlobalData->LightSize->SunLightCount;
	if (CheckHemi) RaysPerVertex += GlobalData->LightSize->HemiLightCount;
	else		   return;

	//�� �����, ��� ������� ������� ������� �� ���, ������� ����� ���� ����� ������. �� ����� ���-�� ����� �� ������ �������, � ����� ������� ���. ����� ��������� id'���� �������� ��������
	u32 CurrentSurfaceID = idx / RaysPerVertex;
	u32 CurrentRayForVertex = idx % RaysPerVertex;

	//that not all. We need determined what type of light we currently
	LightSource LightType = LS_UNKNOWN;
	int LinearLightIndex = 0;
	GetLightTypeAndIndex(GlobalData, CurrentRayForVertex, CheckRGB, CheckSun, CheckHemi, LightType, LinearLightIndex);

	bool IsRayOptimized = GetRayOptimizeOut(GlobalData, RequestedRays, CurrentSurfaceID, LinearLightIndex);

	StatusBuffer[idx] = (char)!IsRayOptimized;
}

// 3
__device__ void GetEnergyFromHit(xrHardwareLCGlobalData* GlobalData, Hit* InHit, float& TargetColor)
{
	TrisAdditionInfo* pTrisInfo = &GlobalData->RaycastModel.TrianglesAdditionInfo[InHit->triId];

	if (!pTrisInfo->CastShadow)
	{
		return;
	}

	xrHardwareTexture& SurfaceTex = GlobalData->Textures[pTrisInfo->TextureID];

	if (!SurfaceTex.IsHaveAlpha)
	{
		TargetColor = 0.0f;
	}
	else
	{
		if (SurfaceTex.Pixels == nullptr)
		{
			//according to GSC source, if the surface has no alpha - we return 0.0f, as all hit passed. But... i dunno
			TargetColor = 0.0f;
			return;
		}

		//#WARNING: Check this code, written based on xrDeflectorLight.cpp :: float getLastRP_Scale(), but not clear understand all parts

		HardwareVector& TexCoords1 = pTrisInfo[0].TexCoords;
		HardwareVector& TexCoords2 = pTrisInfo[1].TexCoords;
		HardwareVector& TexCoords3 = pTrisInfo[2].TexCoords;

		HardwareVector BarycentricCoords(1.0f - InHit->u - InHit->v, InHit->u, InHit->v);

		float FinalU = TexCoords1.x * BarycentricCoords.x + TexCoords2.x * BarycentricCoords.y + TexCoords3.x * BarycentricCoords.z;
		float FinalV = TexCoords1.y * BarycentricCoords.x + TexCoords2.y * BarycentricCoords.y + TexCoords3.y * BarycentricCoords.z;

		int iU = __float2int_rd(FinalU * float(SurfaceTex.Width) + 0.5f);
		int iV = __float2int_rd(FinalV * float(SurfaceTex.Height) + 0.5f);

		iU %= SurfaceTex.Width;			if (iU < 0) iU += SurfaceTex.Width;
		iV %= SurfaceTex.Height;		if (iV < 0) iV += SurfaceTex.Height;

		xrHardwarePixel& PixelData = SurfaceTex.Pixels[iV * SurfaceTex.Width + iU];
		float Opacity = 1.0f - ((float)PixelData.Alpha / 255.0f);

		TargetColor *= Opacity;
	}
}

// 2
__device__ void ShiftRay(xrHardwareLCGlobalData* GlobalData, Ray* CurrentRay, Hit* InHit)
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

	CurrentRay->Origin.Mad_Self(CurrentRay->Direction, Distance + 0.01f);
}

// 1
__global__ void ProcessHits(xrHardwareLCGlobalData* GlobalData, Ray* RayBuffer, Hit* HitBuffer,
	float* ColorBuffer, char* RayStatusBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, bool IsFirstTime,
	bool CheckRGB, bool CheckSun, bool CheckHemi, bool SkipFaceMode, u64* FacesToSkip, u32 * alive_rays_count, int Rounds)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	//early exit condition
	if (idx >= AliveRaysCount) return;

	//second early exit condition. If ray is already done his life
	u32 AliveRayIndex = AliveRaysIndexes[idx];
	char& RayStatus = RayStatusBuffer[AliveRayIndex];

	if (RayStatus == 0) return;

	Hit* OurHit = &HitBuffer[idx];
	Ray* OurRay = &RayBuffer[idx];
	float& OurColor = ColorBuffer[idx];

	if (IsFirstTime)
	{
		OurColor = 1.0f;
	}

	//third early exit condition. If now - first time, and we don't have any hits
	if (OurHit->triId == -1)
	{
		RayStatus = 0;
		OurRay->tmax = 0.0f;
		return;
	}

	//check what current vertex id....
	int RaysPerVertex = 0;
	if (CheckRGB)  RaysPerVertex += GlobalData->LightSize->RGBLightCount;
	if (CheckSun)  RaysPerVertex += GlobalData->LightSize->SunLightCount;
	if (CheckHemi) RaysPerVertex += GlobalData->LightSize->HemiLightCount;

	u32 CurrentSurfaceID = AliveRayIndex / RaysPerVertex;
	u32 CurrentRayForSurface = AliveRayIndex % RaysPerVertex;

	//fouth early exit condition, happens only on FaceToSkip mode
	if (SkipFaceMode)
	{
		TrisAdditionInfo& TriData = GlobalData->RaycastModel.TrianglesAdditionInfo[OurHit->triId];
		if (TriData.FaceID == FacesToSkip[CurrentSurfaceID])
		{
			atomicAdd(&alive_rays_count[0], 1);

			//we hit face, that we must skip. And we do that
			ShiftRay(GlobalData, OurRay, OurHit);
			return;
		}
	}

	//that not all. We need determined what type of light we currently
	LightSource LightType = LS_UNKNOWN;
	int LinearLightIndex = 0;
	GetLightTypeAndIndex(GlobalData, CurrentRayForSurface, CheckRGB, CheckSun, CheckHemi, LightType, LinearLightIndex);

	//get energy from current hit
	GetEnergyFromHit(GlobalData, OurHit, OurColor);

	//we lost all energy, die
	if (OurColor == 0.0f)
	{
		RayStatus = 0;
		OurRay->tmax = 0.0f;
		return;
	}

	atomicAdd(&alive_rays_count[0], 1);  

	//shift ray, so we can cast next...
	ShiftRay(GlobalData, OurRay, OurHit);
}

// COLOR WORK

__device__ void DoFinalizeRay(xrHardwareLCGlobalData* GlobalData, float InputEnergy, RayRequest& Request, R_Light& Light, LightSource InLightType, base_color_c& OutColor)
{
	switch (InLightType)
	{
	case LS_RGB:
	{
		if (Light.type == LT_DIRECT)
		{
			HardwareVector LightDirection = Light.direction.Inverted();

			float DotLight = LightDirection.DotProduct(Request.Normal);

			float CookedEnergy = DotLight * Light.energy * InputEnergy;

			OutColor.rgb.x += CookedEnergy * Light.diffuse.x;
			OutColor.rgb.y += CookedEnergy * Light.diffuse.y;
			OutColor.rgb.z += CookedEnergy * Light.diffuse.z;
		}

		if (Light.type == LT_POINT)
		{
			float SquaredDistance = Request.Position.DistanceSquared(Light.position);

			HardwareVector LightDirection = Light.position.Subtract(Request.Position);
			LightDirection.Normalize_Safe();

			float DotLight = LightDirection.DotProduct(Request.Normal);
			float Distance = sqrtf(SquaredDistance);

			float PreCookedEnergy = DotLight * Light.energy * InputEnergy;

			float CookedEnergy = PreCookedEnergy * (1.0f / (Light.attenuation0 + Light.attenuation1 * Distance + Light.attenuation2 * SquaredDistance) - Distance * Light.falloff);

			OutColor.rgb.x += CookedEnergy * Light.diffuse.x;
			OutColor.rgb.y += CookedEnergy * Light.diffuse.y;
			OutColor.rgb.z += CookedEnergy * Light.diffuse.z;
		}

		if (Light.type == LT_SECONDARY)
		{
			float SquaredDistance = Request.Position.DistanceSquared(Light.position);

			HardwareVector LightDirection = Light.position.Subtract(Request.Position);
			LightDirection.Normalize_Safe();

			float DotLight = LightDirection.DotProduct(Request.Normal);
			DotLight *= -LightDirection.DotProduct(Light.direction);

			float Distance = sqrtf(SquaredDistance);
			float PreCookedEnergy = powf(DotLight, 1.0 / 8.0f) * Light.energy * InputEnergy;
			float CookedEnergy = PreCookedEnergy * (1.0f - Distance / Light.range);

			OutColor.rgb.x += CookedEnergy * Light.diffuse.x;
			OutColor.rgb.y += CookedEnergy * Light.diffuse.y;
			OutColor.rgb.z += CookedEnergy * Light.diffuse.z;
		}
	}
	break;
	case LS_SUN:
	{
		if (Light.type == LT_DIRECT)
		{
			float CookedEnergy = Light.energy * InputEnergy;
			OutColor.sun += CookedEnergy;
		}
		else
		{
			float SquaredDistance = Request.Position.DistanceSquared(Light.position);

			HardwareVector LightDirection = Light.position.Subtract(Request.Position);
			LightDirection.Normalize_Safe();

			float DotLight = LightDirection.DotProduct(Request.Normal);
			float Distance = sqrtf(SquaredDistance);

			float PreCookedEnergy = DotLight * Light.energy * InputEnergy;
			float CookedEnergy = PreCookedEnergy / (Light.attenuation0 + Light.attenuation1 * Distance + Light.attenuation2 * SquaredDistance);
			OutColor.sun += CookedEnergy;
		}
	}
	break;
	case LS_HEMI:
	{
		if (Light.type == LT_DIRECT)
		{
			float CookedEnergy = Light.energy * InputEnergy;
			OutColor.hemi += CookedEnergy;
		}
		else
		{
			float SquaredDistance = Request.Position.DistanceSquared(Light.position);

			HardwareVector LightDirection = Light.position.Subtract(Request.Position);
			LightDirection.Normalize_Safe();

			float DotLight = LightDirection.DotProduct(Request.Normal);
			float Distance = sqrtf(SquaredDistance);

			float PreCookedEnergy = DotLight * Light.energy * InputEnergy;
			float CookedEnergy = PreCookedEnergy / (Light.attenuation0 + Light.attenuation1 * Distance + Light.attenuation2 * SquaredDistance);
			OutColor.hemi += CookedEnergy;
		}
	}
	break;

	case LS_UNKNOWN:
	default:
		break;

	}
}

__global__ void FinalizeRays(xrHardwareLCGlobalData* GlobalData, float* EnergyBuffer, RayRequest* RequestedRays, u32 RequestedRaysCount, base_color_c* OutColorBuffer, u32* AliveRayIndexes, u32 AliveRaysCount, bool CheckRGB, bool CheckSun, bool CheckHemi, int* SurfacePointStartLoc)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	//early exit condition
	if (idx >= RequestedRaysCount) return;

	//idx == SurfaceId
	RayRequest& CurrentRequest = RequestedRays[idx];
	base_color_c& OurFinalColor = OutColorBuffer[idx];

	int RaysPerVertex = 0;
	if (CheckRGB)  RaysPerVertex += GlobalData->LightSize->RGBLightCount;
	if (CheckSun)  RaysPerVertex += GlobalData->LightSize->SunLightCount;
	if (CheckHemi) RaysPerVertex += GlobalData->LightSize->HemiLightCount;

	//calc lower and high bounds
	u32 SurfaceLightsStart = idx * RaysPerVertex;
	u32 SurfaceLightsEnd = SurfaceLightsStart + RaysPerVertex;

	//however, before starting cycle, we need found start point in AliveRayIndexes array
	//#HOTFIX: Use precomputed start location
	u32 AliveRayIndexCursor = SurfacePointStartLoc[idx];

	//next algorithm can do it self, but we need speed up process

	for (; AliveRayIndexCursor < AliveRaysCount; ++AliveRayIndexCursor)
	{
		u32 AliveRay = AliveRayIndexes[AliveRayIndexCursor];

		if (AliveRay < SurfaceLightsEnd)
		{
			u32 LightForSurfaceID = AliveRay % RaysPerVertex;

			LightSource LightType = LS_UNKNOWN;
			int LinearLightIndex = 0;
			GetLightTypeAndIndex(GlobalData, LightForSurfaceID, CheckRGB, CheckSun, CheckHemi, LightType, LinearLightIndex);
			DoFinalizeRay(GlobalData, EnergyBuffer[AliveRayIndexCursor], CurrentRequest, GlobalData->LightData[LinearLightIndex], LightType, OurFinalColor);
		}
		else
		{
			break;
		}
	}

}


// NEW TRIANGLE (NEED BVH SYSTEM TO FASTER WORKING)

#include <assert.h>
 
__device__ void MSG(MsgData* data, char* val, int id)
{
	for (int i = 0; i < 32; i++)
		data->msg[i] = val[i];
	data->value_int = id;
}

__device__ void HitSet(HitCDB* hit, int triID, int u, int v, int dist)
{
	hit->u = u;
	hit->v = v;
	hit->Distance = dist;
	hit->triId = triID;
}

__device__ void __forceinline__ Tri(Model* data, int idx, RayRequest* ray, RcastResult * hit)
{
	
	auto tri = data->tris[idx];
	auto p0 = tri.p1;
	auto p1 = tri.p2;
	auto p2 = tri.p3;

	HardwareVector edge1, edge2, tvec, pvec, qvec;
	float det, inv_det;
	float u, v, range;

	edge1.Subtract(p1, p0);
	edge2.Subtract(p2, p0);
	// begin calculating determinant - also used to calculate U parameter
	// if determinant is near zero, ray lies in plane of triangle
	pvec.CrossProduct(ray->Normal, edge2);
	det = edge1.DotProduct(pvec);

	if (det < 0.0000100f)
		return;

	tvec.Subtract(ray->Position, p0);						// calculate distance from vert0 to ray origin
	u = tvec.DotProduct(pvec);					// calculate U parameter and test bounds

	if (u < 0.f || u > det)
		return;

	qvec.CrossProduct(tvec, edge1);				// prepare to test V parameter
	v = ray->Position.DotProduct(qvec);			// calculate V parameter and test bounds

	if (v < 0.f || u + v > det)
		return;

	range = edge2.DotProduct(qvec);				// calculate t, scale parameters, ray intersects triangle
	inv_det = 1.0f / det;
	range *= inv_det;
	u *= inv_det;
	v *= inv_det;

	hit->result[1].triId = idx;
	hit->result[1].u = u;
	hit->result[1].v = v;
	hit->result[1].Distance = range;
 
}

__global__ void __forceinline__ TriCalculate(int id, Model * data, RayRequest* rays, RcastResult* hits)
{
	//int idx_cuda = blockIdx.x * blockDim.x;
	//int result = 0;			
	//int split = GlobalData->RaycastModel.TrisCount / 1024; 
	//for (int idx = (split * threadIdx.x) - split; idx < split * threadIdx.x; idx++)
	
	int res = threadIdx.x + blockIdx.x * blockDim.x * id;

	for (int idx = 0; idx < data->count; idx++)
	{
		Tri(data, idx, &rays[res], &hits[res]);
	}
}


//CALL

extern "C" cudaError_t RunCheckRayOptimizeOut(xrHardwareLCGlobalData * GlobalData, RayRequest * RequestedRays, char* StatusBuffer, u64 MaxPossibleRays, int flag)
{
	const int BlockSize = 1024;
	int GridSize = ((unsigned int)MaxPossibleRays / BlockSize) + 1;

	//cudaOccupancyMaxPotentialBlockSize(&BlockSize, &GridSize, (void*)CheckRayOptimizeOut, 0, MaxPossibleRays);

	//check flags
	bool IsRGBLightsAllowed = true;
	bool IsHemiLightsAllowed = true;
	bool IsSunLightsAllowed = true;
	if ((flag & LP_dont_rgb) != 0) IsRGBLightsAllowed = false;
	if ((flag & LP_dont_sun) != 0) IsSunLightsAllowed = false;
	if ((flag & LP_dont_hemi) != 0) IsHemiLightsAllowed = false;
	
	CheckRayOptimizeOut << <GridSize, BlockSize >> > (GlobalData, RequestedRays, StatusBuffer, MaxPossibleRays, IsRGBLightsAllowed, IsSunLightsAllowed, IsHemiLightsAllowed);
	return cudaDeviceSynchronize();
}

extern "C" cudaError_t RunGenerateRaysForTask(xrHardwareLCGlobalData * GlobalData, RayRequest * RequestedRays, Ray * RayBuffer, u32 * AliveRaysIndexes, u32 AliveRaysCount, int flag)
{
	int BlockSize = 1024;
	int GridSize = (AliveRaysCount / BlockSize) + 1;

	//check flags
	bool IsRGBLightsAllowed = true;
	bool IsHemiLightsAllowed = true;
	bool IsSunLightsAllowed = true;
	if ((flag & LP_dont_rgb) != 0) IsRGBLightsAllowed = false;
	if ((flag & LP_dont_sun) != 0) IsSunLightsAllowed = false;
	if ((flag & LP_dont_hemi) != 0) IsHemiLightsAllowed = false;

	GenerateRaysForTask <<<GridSize, BlockSize >>> (GlobalData, RequestedRays, RayBuffer, AliveRaysIndexes, AliveRaysCount, IsRGBLightsAllowed, IsSunLightsAllowed, IsHemiLightsAllowed);
	return cudaDeviceSynchronize();
}

extern "C" cudaError_t RunProcessHits(xrHardwareLCGlobalData * GlobalData, Ray * RayBuffer, Hit * HitBuffer, float* ColorBuffer, 
	char* RayStatusBuffer, u32 * AliveRaysIndexes, u32 AliveRaysCount, bool IsFirstTime, int flag, u64 * FacesToSkip, u32 * alive_rays_count, int Rounds)
{
	int BlockSize = 1024;
	int GridSize = (AliveRaysCount / BlockSize) + 1;


	//check flags
	bool IsRGBLightsAllowed = true;
	bool IsHemiLightsAllowed = true;
	bool IsSunLightsAllowed = true;
	if ((flag & LP_dont_rgb) != 0) IsRGBLightsAllowed = false;
	if ((flag & LP_dont_sun) != 0) IsSunLightsAllowed = false;
	if ((flag & LP_dont_hemi) != 0) IsHemiLightsAllowed = false;

	bool SkipFaceMode = !!(flag & LP_UseFaceDisable);

	ProcessHits << <GridSize, BlockSize >> > (
		GlobalData, RayBuffer, HitBuffer, ColorBuffer, 
		RayStatusBuffer, AliveRaysIndexes, AliveRaysCount, IsFirstTime, 
		IsRGBLightsAllowed, IsSunLightsAllowed, IsHemiLightsAllowed, SkipFaceMode, 
		FacesToSkip, alive_rays_count, Rounds);

	return cudaDeviceSynchronize();
}
 
extern "C" cudaError_t RunFinalizeRays(xrHardwareLCGlobalData * GlobalData, float* EnergyBuffer, RayRequest * RequestedRays, u32 RequestedRaysCount, base_color_c * OutColors, u32 * AliveRaysIndexes, u32 AliveRaysCount, int flag, int* SurfacePointStartLoc)
{
	int BlockSize = 1024;
	int GridSize = (RequestedRaysCount / BlockSize) + 1;

	//check flags
	bool IsRGBLightsAllowed = true;
	bool IsHemiLightsAllowed = true;
	bool IsSunLightsAllowed = true;
	if ((flag & LP_dont_rgb) != 0) IsRGBLightsAllowed = false;
	if ((flag & LP_dont_sun) != 0) IsSunLightsAllowed = false;
	if ((flag & LP_dont_hemi) != 0) IsHemiLightsAllowed = false;

	FinalizeRays << <GridSize, BlockSize >> > (GlobalData, EnergyBuffer, RequestedRays, RequestedRaysCount, OutColors, AliveRaysIndexes, AliveRaysCount, IsRGBLightsAllowed, IsSunLightsAllowed, IsHemiLightsAllowed, SurfacePointStartLoc);
	return cudaDeviceSynchronize();
}

//TEST

extern "C" cudaError_t RunTriCollide(Model * model, RayRequest * ray, RcastResult * hits)
{
	int BlockSize = 1024;
	//int GridSize = (RequestedRaysCount / BlockSize) + 1;
	//int GridSize = ( tris_count / BlockSize ) + 1;

	//TriCalculate << <GridSize, BlockSize >> > (GlobalData, ray, hits);

	for (int i = 1; i <= 8; i++)
		TriCalculate << <4, 1024 >> > (i, &model[i - 1], ray, hits);
	//TriCalculate <<<8, 1024 >>> (0, &model[0], ray, hits);

	return cudaDeviceSynchronize();
}

