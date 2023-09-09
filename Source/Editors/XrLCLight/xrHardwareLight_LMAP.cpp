#include "stdafx.h"

#include "xrFaceDefs.h"
#include "xrFace.h"
#include "xrHardwareLight.h"
#include "xrDeflector.h"
#include "xrLC_GlobalData.h"
#include "light_point.h"
#include "base_color.h"

#include "optix_primepp.h"
#include "cuda_runtime.h"

// Include MEM SET

extern bool use_GPU_mem;
extern bool use_GPU_mem_tex;

extern bool use_GPU_mem_stage_1;
extern bool use_GPU_mem_stage_2;
extern bool use_GPU_mem_stage_3;

extern optix::prime::Context PrimeContext;
extern optix::prime::Model LevelModel;



extern "C"
{
	cudaError_t RunCheckRayOptimizeOut(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, char* StatusBuffer, u64 MaxPossibleRays, int flag);
	cudaError_t RunGenerateRaysForTask(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, Ray* RayBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, int flag);
	cudaError_t RunProcessHits(xrHardwareLCGlobalData* GlobalData, Ray* RayBuffer, Hit* HitBuffer, float* ColorBuffer, char* RayStatusBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, bool IsFirstTime, int flag, u64* FacesToSkip, u32* aliverays, int Rounds);
	cudaError_t RunFinalizeRays(xrHardwareLCGlobalData* GlobalData, float* EnergyBuffer, RayRequest* RequestedRays, u32 RequestedRaysCount, base_color_c* OutColors, u32* AliveRaysIndexes, u32 AliveRaysCount, int flag, int* SurfacePointStartLoc);

	cudaError_t RunFinalizeRaysLMAPS(xrHardwareLCGlobalData* GlobalData, float* EnergyBuffer, RayRequest* RequestedRays, u32 RequestedRaysCount, ResultReqvest* OutColors, u32* AliveRaysIndexes, u32 AliveRaysCount, int flag, int* SurfacePointStartLoc);
}


inline void CheckCudaErr(cudaError_t error);
inline DeviceBufferType GetBufferTypeByMode(xrHardwareLight::Mode mode);


void xrHardwareLight::PerformRaycastLMAPS(xr_vector<RayRequestLMAPS>& InRays, int flag, xr_vector<ResultReqvest>& OutHits, bool update_status)
{
	//We use all static light in our calculations for now
	if (InRays.empty())
	{
 		return;
	}

	// LIGHTS
	base_lighting& AllLights = inlc_global_data()->L_static();
	u32 MaxRaysPerPoint = 0;
	if ((flag & LP_dont_rgb) == 0)
	{
		MaxRaysPerPoint += AllLights.rgb.size();
	}
	if ((flag & LP_dont_sun) == 0)
	{
		MaxRaysPerPoint += AllLights.sun.size();
	}
	if ((flag & LP_dont_hemi) == 0)
	{
		MaxRaysPerPoint += AllLights.hemi.size();
	}

	if (MaxRaysPerPoint == 0)
	{
		Msg("! PerformRaycast invoked, but no lights can be accepted");
		Msg("All static lights RGB: %d Sun: %d Hemi: %d", AllLights.rgb.size(), AllLights.sun.size(), AllLights.hemi.size());
		return;
	}

	int cur_split = 0;
	int i = 0;
	xr_map<int, xr_vector<RayRequestLMAPS>> rays_task;
	
	size_t ray_racvest = 0;
	size_t GB = 1024 * 1024 * 1024;
	size_t ray_recvest_max = MaxMem() * GB;

	for (auto ray : InRays)
	{
		if (ray_racvest > ray_recvest_max)
		{
			cur_split++;
			i = 0;
			ray_racvest = 0;
		}

		rays_task[cur_split].push_back(ray);
		i++;
		ray_racvest += (MaxRaysPerPoint * 16);	 //Byte To Allocate MAX ~= 16 
	}

	CTimer t; t.Start();
	for (auto rays : rays_task)
	{
		StatusNoMSG("xrHardwareLight %d/%d", rays.first, rays_task.size());
		//Msg("StartRay: %d", t.GetElapsed_ms());
		xr_vector<ResultReqvest>	hits;
		RaycastingLMAPS(rays.second, flag, hits);
		std::move(hits.begin(), hits.end(), std::back_inserter(OutHits));
	}

	Msg("RayEnd: %d", t.GetElapsed_ms());
}

void xrHardwareLight::RaycastingLMAPS(xr_vector<RayRequestLMAPS>& InRaysLMAP, int flag, xr_vector<ResultReqvest>& OutHits)
{
	xr_vector<RayRequest> InRays;

	for (auto ray : InRaysLMAP)
		InRays.push_back(RayRequest { ray.Position, ray.Normal, ray.FaceToSkip } );
 
	base_lighting& AllLights = inlc_global_data()->L_static();
	u32 MaxRaysPerPoint = 0;
	if ((flag & LP_dont_rgb) == 0)
	{
		MaxRaysPerPoint += AllLights.rgb.size();
	}
	if ((flag & LP_dont_sun) == 0)
	{
		MaxRaysPerPoint += AllLights.sun.size();
	}
	if ((flag & LP_dont_hemi) == 0)
	{
		MaxRaysPerPoint += AllLights.hemi.size();
	}
	size_t MaxPotentialRays = (MaxRaysPerPoint * InRays.size());

	cudaError_t DebugErr = cudaError_t::cudaSuccess;

	auto ZeroOutput = [](int RequestedRays, xr_vector<RayRequestLMAPS>& TargetRays, xr_vector<ResultReqvest>& TargetArray)
	{
		TargetArray.clear();
		TargetArray.reserve(RequestedRays);

		base_color_c ZeroColor;

		for (int i = 0; i < RequestedRays; ++i)
		{
			ResultReqvest data;
			data.C = ZeroColor;
			//data.deflector = TargetRays[i].Deflector;
			//data.U = TargetRays[i].U;
			//data.V = TargetRays[i].V;
			TargetArray.push_back(data);
		}
	};

	auto CopyOutBufferDEFFLECTOR = [](xr_vector<RayRequestLMAPS>& TargetRays, xr_vector<ResultReqvest>& TargetArray)
	{
		base_color_c ZeroColor;

		for (int i = 0; i < TargetRays.size(); ++i)
		{
			TargetArray[i].U = TargetRays[i].U;
			TargetArray[i].V = TargetRays[i].V;
			TargetArray[i].Deflector = TargetRays[i].Deflector;

			/* 
			ResultReqvest data;
			data.C = ZeroColor;
			data.deflector = TargetRays[i].Deflector;
			data.U = TargetRays[i].U;
			data.V = TargetRays[i].V;
			TargetArray.push_back(data);
			*/
		}
	};

	PrintMSG("MEM CPY START %u, Potential Rays = %u", InRays.size() * sizeof(RayRequest) / 1024 / 1024, MaxPotentialRays);

	DeviceBuffer < RayRequest > RayBuffer(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_1, "RayReqvest");
	DebugErr = cudaMemcpy(RayBuffer.ptr(), InRays.data(), InRays.size() * sizeof(RayRequest), cudaMemcpyHostToDevice);			  //
	CheckCudaErr(DebugErr);

	DeviceBuffer<char> RayEnableStatusBuffer(MaxPotentialRays, GetBufferTypeByMode(mode), use_GPU_mem_stage_1, "MaxPotential");
	DebugErr = cudaMemset(RayEnableStatusBuffer.ptr(), 0, MaxPotentialRays);
	CheckCudaErr(DebugErr);

	DebugErr = RunCheckRayOptimizeOut(GlobalData->ptr(), RayBuffer.ptr(), RayEnableStatusBuffer.ptr(), MaxPotentialRays, flag);
	CheckCudaErr(DebugErr);

	const char* StatusBuffer = RayEnableStatusBuffer.hostPtr();
	char* pItStatusBuffer = const_cast <char*> (StatusBuffer);

	CTimer test;
	test.Start();

	xr_vector <int> SurfacePoint2StartLoc;
	xr_vector <u32> AliveRaysIndexes;
	for (u32 Index = 0; Index < MaxPotentialRays; Index++)
	{
		char IsRayAlive = *pItStatusBuffer;
		if (IsRayAlive)
		{
			AliveRaysIndexes.push_back(Index);

			//conditionaly add a surface start point
			u32 SurfaceID = Index / MaxRaysPerPoint;
			u32 RegisteredPoints = SurfacePoint2StartLoc.size();
			if (RegisteredPoints < SurfaceID)
			{
				//wait... we miss a whole surface point?
				//how many surface points missed?
				u32 Differents = SurfaceID - RegisteredPoints;
				//add all missing. Declare it as -1, so we can skip them in FINALIZE stage
				for (int i = 0; i < Differents; ++i)
				{
					SurfacePoint2StartLoc.push_back(-1);
				}
			}
			if (RegisteredPoints == SurfaceID)
			{
				SurfacePoint2StartLoc.push_back((int)AliveRaysIndexes.size() - 1);
			}
		}
		pItStatusBuffer++;
	}

	if (AliveRaysIndexes.empty())
	{
		//all rays are optimized
		ZeroOutput(InRays.size(), InRaysLMAP, OutHits);
		return;
	}

	//create rays buffer and fill them through cuda
	DeviceBuffer <u32> DeviceAliveRaysIndexes(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2, "AliveRaysIndexer");
	DebugErr = cudaMemcpy(DeviceAliveRaysIndexes.ptr(), AliveRaysIndexes.data(), AliveRaysIndexes.size() * sizeof(u32), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);

	DeviceBuffer <int> DeviceSurfacePoint2StartLoc(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2, "DeviceSurfacePoint2StartLoc");
	DebugErr = cudaMemcpy(DeviceSurfacePoint2StartLoc.ptr(), SurfacePoint2StartLoc.data(), SurfacePoint2StartLoc.size() * sizeof(int), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);

	//PrintMSG("MemSet RAYS: %u", AliveRaysIndexes.size() * sizeof(Ray) / 1024 / 1024);

	DeviceBuffer<Ray> OptimizedRaysVec(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2, "Ray");  //use_GPU_mem
	DebugErr = RunGenerateRaysForTask(GlobalData->ptr(), RayBuffer.ptr(), OptimizedRaysVec.ptr(), DeviceAliveRaysIndexes.ptr(), AliveRaysIndexes.size(), flag);
	CheckCudaErr(DebugErr);

	//PrintMSG("MemSet HITS: %u", AliveRaysIndexes.size() * sizeof(Hit) / 1024 / 1024);

	DeviceBuffer <Hit> OptimizedHitsVec(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2, "Hit");
	DebugErr = cudaMemset(OptimizedHitsVec.ptr(), 0, OptimizedHitsVec.count() * sizeof(Hit));
	CheckCudaErr(DebugErr);

	//Msg("Hits: %d", OptimizedHitsVec.count());
	//Msg("Rays: %d", OptimizedRaysVec.count());

	optix::prime::BufferDesc OptmizedHitDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Hit::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedHitsVec.ptr());
	OptmizedHitDescBuffer->setRange(0, OptimizedHitsVec.count());

	optix::prime::BufferDesc OptimizedRaysDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Ray::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedRaysVec.ptr());
	OptimizedRaysDescBuffer->setRange(0, OptimizedRaysVec.count());

	optix::prime::Query LevelQuery = LevelModel->createQuery(RTP_QUERY_TYPE_CLOSEST);
	LevelQuery->setHits(OptmizedHitDescBuffer);
	LevelQuery->setRays(OptimizedRaysDescBuffer);


	//if "skip face" mode enabled - load special buffer for every requested surface point
	bool SkipFaceMode = !!(flag & LP_UseFaceDisable);
	DeviceBuffer <u64> FaceToSkip(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3, "SkipFaces");

	//go go round system!

	u32 AliveRays = AliveRaysIndexes.size();
	DeviceBuffer<float> RayEnergy(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3, "AliveRays");
	bool IsFirstCall = true;
	int Rounds = 0;

	u64* FacesToSkip = nullptr;
	if (SkipFaceMode)
	{
		xr_vector <u64> HostFaceToSkip; HostFaceToSkip.reserve(InRays.size());
		for (RayRequest& InRay : InRays)
			HostFaceToSkip.push_back((u64)InRay.FaceToSkip);

		DebugErr = cudaMemcpy(FaceToSkip.ptr(), HostFaceToSkip.data(), HostFaceToSkip.size() * sizeof(u64), cudaMemcpyHostToDevice);
		CheckCudaErr(DebugErr);
		HostFaceToSkip.clear();

		FacesToSkip = FaceToSkip.ptr();
	}

	u32 CPU_USED = 0;
	u32 GPU_USED = 0;

	DeviceBuffer<u32>  AliveCounts(1024);

	while (AliveRays)
	{
		CTimer tt; tt.Start();
		LevelQuery->execute(0);

		AliveRays = 0;
		CPU_USED += tt.GetElapsed_ms();

		tt.Start();

		DebugErr = RunProcessHits(GlobalData->ptr(),
			OptimizedRaysVec.ptr(),
			OptimizedHitsVec.ptr(),
			RayEnergy.ptr(),
			RayEnableStatusBuffer.ptr(),
			DeviceAliveRaysIndexes.ptr(),
			AliveRaysIndexes.size(),
			IsFirstCall,
			flag,
			FacesToSkip,
			AliveCounts.ptr(),
			Rounds);
		CheckCudaErr(DebugErr);

		int rays = AliveCounts.hostPtr()[0];
		AliveRays = rays;

		GPU_USED += tt.GetElapsed_ms();

		if (IsFirstCall)
			IsFirstCall = false;
		++Rounds;

		u32 zero = 0;
		AliveCounts.copyToBuffer(&zero, 128);
	} 

	Msg("CPU ms:%d, GPU ms: %d", CPU_USED, GPU_USED);

	/// OTHER
	ZeroOutput(InRays.size(), InRaysLMAP, OutHits);

	DeviceBuffer <ResultReqvest> FinalColorBuffer(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3, "base_color_c");
	DebugErr = cudaMemcpy(FinalColorBuffer.ptr(), OutHits.data(), OutHits.size() * sizeof(ResultReqvest), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);

	DebugErr = RunFinalizeRaysLMAPS(
		GlobalData->ptr(),
		RayEnergy.ptr(),
		RayBuffer.ptr(),
		RayBuffer.count(),
		FinalColorBuffer.ptr(),
		DeviceAliveRaysIndexes.ptr(),
		DeviceAliveRaysIndexes.count(),
		flag,
		DeviceSurfacePoint2StartLoc.ptr());
	CheckCudaErr(DebugErr);

	//copy directly back
	DebugErr = cudaMemcpy(OutHits.data(), FinalColorBuffer.ptr(), FinalColorBuffer.count() * sizeof(ResultReqvest), cudaMemcpyDeviceToHost);
	CheckCudaErr(DebugErr);

	CopyOutBufferDEFFLECTOR(InRaysLMAP, OutHits);
}