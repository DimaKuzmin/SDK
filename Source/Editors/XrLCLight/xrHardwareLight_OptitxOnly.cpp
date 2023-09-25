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

optix::prime::Context	 PrimeContext_Optix;

//level buffers
optix::prime::BufferDesc LevelIndixes_Optix;
optix::prime::BufferDesc LevelVertexes_Optix;
optix::prime::Model		 LevelModel_Optix;

DeviceBuffer<HardwareVector> CDBVertexesBuffer_Optix;
DeviceBuffer<PolyIndexes>	 CDBTrisIndexBuffer_Optix;

void LoadLevelOptix()
{
	CDB::MODEL* model = lc_global_data()->RCAST_Model();

	CDBVertexesBuffer_Optix.alloc(model->get_verts_count(), DeviceBufferType::CUDA, true);
	CDBVertexesBuffer_Optix.copyToBuffer(reinterpret_cast<HardwareVector*>(model->get_verts()), model->get_verts_count());

	{
	  	size_t freeBytes, totalBytes;
		cudaMemGetInfo(&freeBytes, &totalBytes);
		Msg("Load To GPU CDB_Vertex: free: %llu, total: %llu", freeBytes / 1024 / 1024, totalBytes / 1024 / 1024);
	}


	xr_vector <PolyIndexes> OptimizedMeshTris;
	OptimizedMeshTris.reserve(model->get_tris_count());

	for (int i = 0; i < model->get_tris_count(); i++)
	{
		CDB::TRI Tris = model->get_tris()[i];
		PolyIndexes indx{ (u32)Tris.verts[0], (u32)Tris.verts[1], (u32)Tris.verts[2] };
		OptimizedMeshTris.push_back(indx);
	}

	CDBTrisIndexBuffer_Optix.alloc(OptimizedMeshTris.size(), DeviceBufferType::CUDA, true);
	CDBTrisIndexBuffer_Optix.copyToBuffer(OptimizedMeshTris.data(), OptimizedMeshTris.size());

	{
	  	size_t freeBytes, totalBytes;
		cudaMemGetInfo(&freeBytes, &totalBytes);
		Msg("Load To GPU  CDB_Tris: free: %llu, total: %llu", freeBytes / 1024 / 1024, totalBytes / 1024 / 1024);
	}

	//Optix Load
	PrimeContext_Optix = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    	
	//OPTIX INIT
	LevelIndixes_Optix = PrimeContext_Optix->createBufferDesc(RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBTrisIndexBuffer_Optix.ptr());
	LevelIndixes_Optix->setRange(0, CDBTrisIndexBuffer_Optix.count());
	
	{
	  	size_t freeBytes, totalBytes;
		cudaMemGetInfo(&freeBytes, &totalBytes);
		Msg("Load To Optix Tris: free: %llu, total: %llu", freeBytes / 1024 / 1024, totalBytes / 1024 / 1024);
	}

	LevelVertexes_Optix = PrimeContext_Optix->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBVertexesBuffer_Optix.ptr());
	LevelVertexes_Optix->setRange(0, CDBVertexesBuffer_Optix.count());

	{
		size_t freeBytes, totalBytes;
		cudaMemGetInfo(&freeBytes, &totalBytes);
		Msg("Load To Optix Vertex: free: %llu, total: %llu", freeBytes / 1024 / 1024, totalBytes / 1024 / 1024);
	}


	LevelModel_Optix = PrimeContext_Optix->createModel();
	LevelModel_Optix->setTriangles(LevelIndixes_Optix, LevelVertexes_Optix);
	LevelModel_Optix->update(0);
}

void UnLoadLevelOptix()
{
	 CDBVertexesBuffer_Optix.free();
	 CDBTrisIndexBuffer_Optix.free();
}

inline void CheckCudaErr(cudaError_t error);


void GetLightsForOPTIX(LightpointRequest& reqvest, std::vector<Ray>& rays_buffer, std::vector<RayAdditionalData>& additional, int flag, size_t& BUFFER_ID) 
{
	base_lighting& lights = lc_global_data()->L_static();
	/*
		HardwareVector Origin;
		float tmin;
		HardwareVector Direction;
		float tmax;
	*/

	Fvector		Ldir, Pnew;
	Pnew.mad(reqvest.Position, reqvest.Normal, 0.01f);
	
	if ((flag & LP_dont_hemi) == 0)
	{
		R_Light* L = &*lights.hemi.begin(), * E = &*lights.hemi.end();

		for (; L != E; L++)
		{
			if (L->type == LT_DIRECT)
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(reqvest.Normal);
				if (D <= 0) continue;
				 
				// Trace Light
				Fvector		PMoved;
				PMoved.mad(Pnew, Ldir, 0.001f);
			  
				Ray ray;
				ray.tmin = 0.f;
				ray.tmax = 1000.0f;
				ray.Origin = PMoved;
				ray.Direction = Ldir;
				rays_buffer[BUFFER_ID] = ray;

				RayAdditionalData data;
				data.U = reqvest.X;
				data.V = reqvest.Y;
				data.skip = (Face*) reqvest.FaceToSkip;
				data.L = L;
				data.type_light = RLightTypes::HEMI;
				additional[BUFFER_ID] = data;
				

				BUFFER_ID++;
			}
			else
			{
				Fvector pos = L->position;

				// Distance
				float sqD = reqvest.Position.distance_to_sqr(pos);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, reqvest.Position);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(reqvest.Normal);
				if (D <= 0) continue;

				float Range = _sqrt(sqD);

				Ray ray;
				ray.tmin = 0.f;
				ray.tmax = Range;
				ray.Origin = Pnew;
				ray.Direction = Ldir;
				rays_buffer[BUFFER_ID] = ray;
								
				RayAdditionalData data;
				data.U = reqvest.X;
				data.V = reqvest.Y;
				data.skip = (Face*) reqvest.FaceToSkip;
				data.L = L;
				data.type_light = RLightTypes::HEMI;
				data.DotProduct = D;
 				data.SquaredDir = sqD;
				additional[BUFFER_ID] = data;

				BUFFER_ID++;
			}
		}
	}

	if ((flag & LP_dont_rgb) == 0)
	{
		R_Light* L = &*lights.rgb.begin(), * E = &*lights.rgb.end();
		for (; L != E; L++)
		{
			switch (L->type)
			{
				case LT_DIRECT:
				{
					// Cos
					Ldir.invert(L->direction);
					float D = Ldir.dotproduct(reqvest.Normal);
					if (D <= 0) continue;
 
					Ray ray;
					ray.Origin = Pnew;
					ray.Direction = Ldir;
					ray.tmax = 1000.0f;
					ray.tmin = 0.f;
					rays_buffer.push_back(ray);

					RayAdditionalData data;
					data.U = reqvest.X;
					data.V = reqvest.Y;
					data.skip = (Face*) reqvest.FaceToSkip;
					data.L = L;
					data.type_light = RLightTypes::RGB;
					data.DotProduct = D;
					additional[BUFFER_ID] = data;

					BUFFER_ID++;
				}
				break;
				case LT_POINT:
				{
					// Distance
					float sqD = reqvest.Position.distance_to_sqr(L->position);
					if (sqD > L->range2) continue;

					// Dir
					Ldir.sub(L->position, reqvest.Position);
					Ldir.normalize_safe();
					float D = Ldir.dotproduct(reqvest.Normal);
					if (D <= 0)			continue;
			    
					// Trace Light
					float Range = _sqrt(sqD);
					
					Ray ray;
					ray.Origin = Pnew;
					ray.Direction = Ldir;
					ray.tmax = Range;
					ray.tmin = 0.f;
					rays_buffer[BUFFER_ID] = ray;

					RayAdditionalData data;
					data.U = reqvest.X;
					data.V = reqvest.Y;
					data.skip = (Face*) reqvest.FaceToSkip;
					data.L = L;
 					data.type_light = RLightTypes::RGB;
					data.DotProduct = D;
 					data.SquaredDir = sqD;
					additional[BUFFER_ID] = data;

					BUFFER_ID++;
				}
				break;
				case LT_SECONDARY:
				{
					// Distance
					float sqD = reqvest.Position.distance_to_sqr(L->position);
					if (sqD > L->range2) continue;

					// Dir
					Ldir.sub(L->position, reqvest.Position);
					Ldir.normalize_safe();
					float	D = Ldir.dotproduct(reqvest.Normal);
					if (D <= 0) continue;
					D *= -Ldir.dotproduct(L->direction);
					if (D <= 0) continue;
	
					// Jitter + trace light -> monte-carlo method
 					Fvector	Psave = L->position, Pdir;
					L->position.mad(Pdir.random_dir(L->direction, PI_DIV_4), .05f);
					float R = _sqrt(sqD);
			
					Ray ray;
					ray.Origin = Pnew;
					ray.Direction = Ldir;
					ray.tmax = R;
					ray.tmin = 0.f;
					rays_buffer[BUFFER_ID] = (ray); 

					
					RayAdditionalData data;
					data.U = reqvest.X;
					data.V = reqvest.Y;
					data.skip = (Face*) reqvest.FaceToSkip;
					data.L = L;
					data.type_light = RLightTypes::RGB;
					data.DotProduct = D;
 					data.SquaredDir = sqD;
					additional[BUFFER_ID] = data;

					BUFFER_ID++;
				}
				break;
			}
		}
	}
 
	if ((flag & LP_dont_sun) == 0)
	{
		R_Light* L = &*lights.sun.begin(), * E = &*lights.sun.end();

		for (; L != E; L++)
		{
			if (L->type == LT_DIRECT)
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(reqvest.Normal);
				if (D <= 0) continue;

				// Trace Light
 				Ray ray;
				ray.Origin = Pnew;
				ray.Direction = Ldir;
				ray.tmax = 1000.f;
				ray.tmin = 0.f;
				rays_buffer[BUFFER_ID] = (ray);		

				
				RayAdditionalData data;
				data.U = reqvest.X;
				data.V = reqvest.Y;
				data.skip = (Face*) reqvest.FaceToSkip;
				data.L = L;
				data.type_light = RLightTypes::SUN;
				additional[BUFFER_ID] = (data);

				BUFFER_ID++;
			}
			else
			{
				// Distance
				float sqD = reqvest.Position.distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, reqvest.Position);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(reqvest.Normal);
				if (D <= 0)			continue;

		
				// Trace Light			
				float Range = _sqrt(sqD);
				
				Ray ray;
				ray.Origin = Pnew;
				ray.Direction = Ldir;
				ray.tmax = Range;
				ray.tmin = 0.f;
				rays_buffer[BUFFER_ID] = (ray);	

				
				RayAdditionalData data;
				data.U = reqvest.X;
				data.V = reqvest.Y;
				data.skip = (Face*) reqvest.FaceToSkip;
				data.L = L;
				data.type_light = RLightTypes::SUN;
				data.DotProduct = D;
 				data.SquaredDir = sqD;
				additional[BUFFER_ID] = (data);

				BUFFER_ID++;
			}	
		}
	}
}


u32 GPU_TIME_BUFFERS = 0;
u32 GPU_TIME_WHILE = 0;
u32 GPU_TIME = 0;
u32 GPU_TIME_PROCESS = 0;
u32 GPU_TIME_MEMCPY = 0;
 
CTimer DeviceTimer;
u32 preDevice = 0;
bool DeviceTimerSetuped = false;

extern "C"
{
	cudaError_t RunProcessHits_NO_TEXTURE(Ray* RayBuffer, Hit* HitBuffer, /*Hit256* HitsData,*/ int* alive_rays, size_t size_hits);
};


void RaycastOptix(std::vector<Ray>& rays, std::vector<HitsResultVector>& outbuffer, size_t BUFFER_ID_MAX)
{
	CTimer total;

	total.Start();
	
	DeviceBuffer<Ray> OptimizedRaysVec(rays.size(), DeviceBufferType::CUDA, true, "Ray");  //use_GPU_mem
	CheckCudaErr( cudaMemcpy( OptimizedRaysVec.ptr(), rays.data(), rays.size() * sizeof(Ray), cudaMemcpyHostToDevice ) );
 
	DeviceBuffer <Hit> OptimizedHitsVec(rays.size(), DeviceBufferType::CUDA, true, "Hit");
	CheckCudaErr( cudaMemset( OptimizedHitsVec.ptr(), 0, OptimizedHitsVec.count() * sizeof(Hit) ) );

	//DeviceBuffer <Hit256> OptimizedHitsVec256(rays.size(), DeviceBufferType::CUDA, true, "Hit256");
	//CheckCudaErr( cudaMemset( OptimizedHitsVec256.ptr(), 0, OptimizedHitsVec256.count() * sizeof(Hit256) ) );
   

	optix::prime::BufferDesc OptimizedRaysDescBuffer = PrimeContext_Optix->createBufferDesc((RTPbufferformat)Ray::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedRaysVec.ptr());
	OptimizedRaysDescBuffer->setRange(0, OptimizedRaysVec.count());

	optix::prime::BufferDesc OptmizedHitDescBuffer = PrimeContext_Optix->createBufferDesc((RTPbufferformat)Hit::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedHitsVec.ptr());
	OptmizedHitDescBuffer->setRange(0, OptimizedHitsVec.count());

	optix::prime::Query LevelQuery = LevelModel_Optix->createQuery(RTP_QUERY_TYPE_CLOSEST);
	LevelQuery->setRays(OptimizedRaysDescBuffer);
	LevelQuery->setHits(OptmizedHitDescBuffer);

	DeviceBuffer<int>  AliveCounts(1);
	int zero = 0;
	//AliveCounts.copyToBuffer(&zero, 1);
	CheckCudaErr( cudaMemset( AliveCounts.ptr(), 0, sizeof(int) * AliveCounts.count() ) );

	int alive_rays = OptimizedRaysVec.count();

	
	outbuffer.resize(OptimizedHitsVec.count());

	std::vector<Hit> result; result.resize(OptimizedHitsVec.count());
	int IDS = 0;

	while(alive_rays)
	{
		IDS++;
		
		if (IDS > 32)
			break;
		
		Msg("WHILE: %d", IDS);
		alive_rays = 0;
	   		
		CTimer tRQ;
		tRQ.Start();
		LevelQuery->execute(0);
		GPU_TIME += tRQ.GetElapsed_ms();
		
		tRQ.Start();
		CheckCudaErr( RunProcessHits_NO_TEXTURE(OptimizedRaysVec.ptr(), OptimizedHitsVec.ptr(), /*OptimizedHitsVec256.ptr(),*/ AliveCounts.ptr(), BUFFER_ID_MAX) );
		GPU_TIME_PROCESS += tRQ.GetElapsed_ms();  
		tRQ.Start();
 
	    
		CheckCudaErr( cudaMemcpy(result.data(), OptimizedHitsVec.ptr(), OptimizedHitsVec.count() * sizeof(Hit), cudaMemcpyDeviceToHost) );
		alive_rays = AliveCounts.hostPtr()[0]; 
		 
		GPU_TIME_MEMCPY += tRQ.GetElapsed_ms();

 		tRQ.Start();

		//auto Vector = OptimizedHitsVec.hostPtrVec();

		for (auto it = 0; it < result.size(); it++)
		{
			const Hit& hit = result[it];
			
			if (hit.triId != -1)
			{
				outbuffer[it].hits.push_back(hit);
				outbuffer[it].count++;
			}
		}

		GPU_TIME_BUFFERS += tRQ.GetElapsed_ms();

		CheckCudaErr( cudaMemset( AliveCounts.ptr(), 0, sizeof(int) * AliveCounts.count() ) );

 	}
 
	//tCUDA.Start();
	//CheckCudaErr( cudaMemcpy(outbuffer.data(), OptimizedHitsVec256.ptr(), BUFFER_ID_MAX * sizeof(Hit256), cudaMemcpyDeviceToHost) );
	//GPU_TIME_MEMCPY+=tCUDA.GetElapsed_ms();

	if (!DeviceTimerSetuped)
	{
		DeviceTimer.Start();
		DeviceTimerSetuped = true;
	}


	if (DeviceTimer.GetElapsed_sec() > preDevice)
	{
		preDevice = DeviceTimer.GetElapsed_sec() + 20;
		Msg("COPY_TO_OPTIX: %u, GPU: %u, GPU_OPTIX: %u, GPU_CUDA: %u, GPU_MEMCPY: %u", 
			GPU_TIME_BUFFERS ,GPU_TIME_WHILE, GPU_TIME, GPU_TIME_PROCESS, GPU_TIME_MEMCPY);
		//GPU_TIME_WHILE = 0;
		//GPU_TIME = 0;
		//GPU_TIME_PROCESS = 0;
		//GPU_TIME_MEMCPY = 0;
		//GPU_MEMCPY = 0;
	}

	GPU_TIME_WHILE += total.GetElapsed_ms();


	//xr_vector<Hit256> hits_result;
	//CheckCudaErr( cudaMemcpy(hits_result.data(), OptimizedHitsVec256.ptr(), OptimizedHitsVec256.count(), cudaMemcpyDeviceToHost) );


}



