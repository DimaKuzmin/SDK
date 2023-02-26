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


optix::prime::Context PrimeContext;

//level buffers
optix::prime::BufferDesc LevelIndixes;
optix::prime::BufferDesc LevelVertexes;
optix::prime::Model LevelModel;

xr_vector <PolyIndexes> LevelIndices;
xr_vector <HardwareVector> LevelVertexData;

xr_vector<Fvector> CPURayVertexData;
xr_vector<Fvector> GPURayVertexData;


///#REFACTOR
//static buffer declaration, replace with pimpl paradigm when refactor time happen
DeviceBuffer<HardwareVector> CDBVertexesBuffer;
DeviceBuffer<PolyIndexes> CDBTrisIndexBuffer;
DeviceBuffer<TrisAdditionInfo> CDBTrisAdditionBuffer;



int cpuidx = -1;
int gpuidx = -1;
unsigned long long OpenCLFreeMemHack = 0;

extern "C"
{
	cudaError_t RunCheckRayOptimizeOut(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, char* StatusBuffer, u64 MaxPossibleRays, int flag);
	cudaError_t RunGenerateRaysForTask(xrHardwareLCGlobalData* GlobalData, RayRequest* RequestedRays, Ray* RayBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, int flag);
	cudaError_t RunProcessHits(xrHardwareLCGlobalData* GlobalData, Ray* RayBuffer, Hit* HitBuffer, float* ColorBuffer, char* RayStatusBuffer, u32* AliveRaysIndexes, u32 AliveRaysCount, bool IsFirstTime, int flag, u64* FacesToSkip);
	cudaError_t RunFinalizeRays(xrHardwareLCGlobalData* GlobalData, float* EnergyBuffer, RayRequest* RequestedRays, u32 RequestedRaysCount, base_color_c* OutColors, u32* AliveRaysIndexes, u32 AliveRaysCount, int flag, int* SurfacePointStartLoc);
}


inline DeviceBufferType GetBufferTypeByMode(xrHardwareLight::Mode mode)
{
	switch (mode)
	{
		case xrHardwareLight::Mode::CUDA:
			return DeviceBufferType::CUDA;
 		default:
			return DeviceBufferType::CUDA;
	}
}

inline void CheckCudaErr(cudaError_t error)
{
	if (error != cudaError_t::cudaSuccess)
	{
		Msg("error: %d", error);
		DebugBreak();
	}
}
							  

BOOL xrHardwareLight::IsHardwareAccelerationSupported(xrHardwareLight::Mode mode)
{
	switch (mode)
	{
		case xrHardwareLight::Mode::CUDA:
		{
			RTPcontext Context;
			RTPresult Result = rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &Context);

			if (Result == RTP_SUCCESS)
			{
				rtpContextDestroy(Context);
				return TRUE;
			}
			return FALSE;
		}
		default:
			return FALSE;
	}


}

xrHardwareLight& xrHardwareLight::Get()
{
	static xrHardwareLight MainSingleton;
	return MainSingleton;
}

bool xrHardwareLight::IsEnabled()
{
	return _IsEnabled;
}

xrHardwareLight::xrHardwareLight() :
	LightBuffer(nullptr),
	LightSizeBuffer(nullptr),
	TrisBuffer(nullptr),
	VertBuffer(nullptr),
	TextureBuffer(nullptr),
	GlobalData(nullptr),
	VertNormalBuffer(nullptr)
{
	mode = Mode::CUDA;

	BOOL bIsCUDASupported = IsHardwareAccelerationSupported(Mode::CUDA);
	BOOL bIsOpenCLSupported = IsHardwareAccelerationSupported(Mode::OpenCL);
	BOOL bIsCPUAccelerationSupported = IsHardwareAccelerationSupported(Mode::CPU);

	// We preffer CUDA, then OpenCL, CPU
	if (bIsCUDASupported)
	{
		mode = Mode::CUDA;
	}	    
	
	//Msg(" [xrHardwareLight]: %s mode", ToString_Mode(mode));

	switch (mode)
	{
		case xrHardwareLight::Mode::CUDA:
			PrimeContext = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
			DeviceMemoryForLevel = 0;
			break;
		default:
			break;
	}

	LightSizeBuffer = new DeviceBuffer<LightSizeInfo>(1, GetBufferTypeByMode(mode));
}

xrHardwareLight::~xrHardwareLight()
{
	if (LightBuffer != nullptr)		delete LightBuffer;
	if (LightSizeBuffer != nullptr)		delete LightSizeBuffer;
	if (TrisBuffer != nullptr)		delete TrisBuffer;
	if (VertBuffer != nullptr)		delete VertBuffer;
	if (TextureBuffer != nullptr)		delete TextureBuffer;
	if (GlobalData != nullptr)		delete GlobalData;
	if (VertNormalBuffer != nullptr)	delete VertNormalBuffer;

	CDBVertexesBuffer.free();
	CDBTrisIndexBuffer.free();
	CDBTrisAdditionBuffer.free();		  
}

xr_map<int, xr_vector<RayRequest>> cur_task;


void xrHardwareLight::PerformRaycast1024(xr_vector<RayRequest>& InRays, int flag, xr_vector<base_color_c>& OutHits)
{
//	PerformRaycast(InRays, flag, OutHits);

//	return;
	cur_task.clear();

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

 
	int cur_split = 0;
	int id = 0;
	for (auto ray : InRays)
	{
		RayRequest rc = ray;
 		cur_task[cur_split].push_back(rc);

		//Msg("POS [%f][%f][%f]", VPUSH(rc.Position));

		if (id > 120000*4)
		{
			cur_split++;
			id = 0;
		}
		id++;
 	}

	//Msg("Start RayCast %d/%d", cur_task.size(), InRays.size());
	
	xr_vector<base_color_c> out;

 
	for (auto map : cur_task)
	{
		Msg("Start RayCast %d/%d", map.first, cur_task.size());

		PerformRaycast(map.second, flag, out);

		for (auto color : out)
			OutHits.push_back(color);
	}
	 
}

bool use_GPU_mem = true;
bool use_GPU_mem_tex = true;

bool use_GPU_mem_stage_1 = true;
bool use_GPU_mem_stage_2 = true;
bool use_GPU_mem_stage_3 = true;



void xrHardwareLight::LoadLevel(CDB::MODEL* RaycastModel, base_lighting& Lightings, xr_vector<b_BuildTexture>& Textures)
{
	Progress(0.0f);
	_IsEnabled = true;
	vecFace& Polygons = inlc_global_data()->g_faces();

	// Check if we have enough video RAM to proceed loading
	size_t LevelMemoryRequired = GetMemoryRequiredForLoadLevel(RaycastModel, Lightings, Textures);
	size_t DeviceFreeMemory = GetDeviceFreeMem();
	if (LevelMemoryRequired > DeviceFreeMemory)
	{
		string256 m;
		xr_sprintf(m, "Build failed!\nFree video RAM (%zu MB) not enough to load level (%zu MB)", DeviceFreeMemory / 1024 / 1024, LevelMemoryRequired / 1024 / 1024);
		Msg(m);
		ExitProcess(1);
		return;
	}

	//###LEVEL GEOMETRY

	//load CDB::MODEL and original model
	//LOAD CDB::MODEL
	HardwareModel OptimizedModel;

	//vertexes
	CDBVertexesBuffer.alloc(RaycastModel->get_verts_count(), GetBufferTypeByMode(mode), use_GPU_mem);
	CDBVertexesBuffer.copyToBuffer(reinterpret_cast<HardwareVector*>(RaycastModel->get_verts()), RaycastModel->get_verts_count());
	OptimizedModel.Vertexes = CDBVertexesBuffer.ptr();
	OptimizedModel.VertexCount = CDBVertexesBuffer.count();
	OptimizedModel.VertexNormal = nullptr;
	
	//normals
	//skip normals no way to retrieve it from optimized model
	//hmm..... 
	//RaycastModel->get_tris()->
	xr_vector <PolyIndexes> OptimizedMeshTris;
	OptimizedMeshTris.reserve(RaycastModel->get_tris_count());

	xr_vector<TrisAdditionInfo> OptimizedTrisAdditionInfo;
	OptimizedTrisAdditionInfo.reserve(RaycastModel->get_tris_count());
	for (int i = 0; i < RaycastModel->get_tris_count(); i++)
	{
		CDB::TRI Tris = RaycastModel->get_tris()[i];
		PolyIndexes indx{ (u32)Tris.verts[0], (u32)Tris.verts[1], (u32)Tris.verts[2] };

		OptimizedMeshTris.push_back(indx);

		TrisAdditionInfo AdditionInfo;

		base_Face& FaceRef = *(base_Face*) Tris.pointer;

		const Shader_xrLC& TrisShader = FaceRef.Shader();

		AdditionInfo.CastShadow = !!TrisShader.flags.bLIGHT_CastShadow;

		Fvector2* pCDBTexCoord = FaceRef.getTC0();
		AdditionInfo.TexCoords = *pCDBTexCoord;

		b_material& TrisMaterial = inlc_global_data()->materials()[FaceRef.dwMaterial];
		AdditionInfo.TextureID = TrisMaterial.surfidx;

		//#WARNING: :(
		AdditionInfo.FaceID = (u64)&FaceRef;
		OptimizedTrisAdditionInfo.push_back(AdditionInfo);
	}

	CDBTrisIndexBuffer.alloc(OptimizedMeshTris.size(), GetBufferTypeByMode(mode), use_GPU_mem);
	CDBTrisIndexBuffer.copyToBuffer(OptimizedMeshTris.data(), OptimizedMeshTris.size());

	CDBTrisAdditionBuffer.alloc(OptimizedTrisAdditionInfo.size(), GetBufferTypeByMode(mode), use_GPU_mem);
	CDBTrisAdditionBuffer.copyToBuffer(OptimizedTrisAdditionInfo.data(), OptimizedTrisAdditionInfo.size());

	OptimizedModel.Tris = CDBTrisIndexBuffer.ptr();
	OptimizedModel.TrisCount = OptimizedMeshTris.size();
	OptimizedModel.TrianglesAdditionInfo = CDBTrisAdditionBuffer.ptr();
	OptimizedModel.TrisAditinalInfoCount = OptimizedTrisAdditionInfo.size();

	Progress(0.1f);

	//ORIGINAL MODEL
#if 0
	HardwareModel OriginalModel;

	vecVertex& LevelVertexesData = inlc_global_data()->g_vertices();

	GetLevelIndices(LevelVertexesData, Polygons, LevelIndices, LevelVertexData);

	//at this moment our Progress bar is on 0.6
	TrisBuffer = new Buffer<PolyIndexes>(LevelIndices.size(), RTP_BUFFER_TYPE_CUDA_LINEAR, UNLOCKED);
	VertBuffer = new Buffer<Fvector>(LevelVertexesData.size(), RTP_BUFFER_TYPE_CUDA_LINEAR, UNLOCKED);
	VertNormalBuffer = new Buffer<HardwareVector>(LevelVertexesData.size(), RTP_BUFFER_TYPE_CUDA_LINEAR, UNLOCKED);

	DebugErr = cudaMemcpy(TrisBuffer->ptr(), LevelIndices.data(), LevelIndices.size() * sizeof(PolyIndexes), cudaMemcpyHostToDevice);
	DebugErr = cudaMemcpy(VertBuffer->ptr(), LevelVertexData.data(), LevelVertexData.size() * sizeof(HardwareVector), cudaMemcpyHostToDevice);
	Progress(0.65f);

	//load Vertex normals
	xr_vector <HardwareVector> VertexNormals;
	VertexNormals.reserve(LevelVertexesData.size());

	for (Vertex* Vert : LevelVertexesData)
	{

		HardwareVector VertexNormal = Vert->N;
		VertexNormals.push_back(VertexNormal);
	}

	DebugErr = cudaMemcpy(VertNormalBuffer->ptr(), VertexNormals.data(), VertexNormals.size() * sizeof(HardwareVector), cudaMemcpyHostToDevice);

	OriginalModel.Tris = TrisBuffer->ptr();
	OriginalModel.Vertexes = (HardwareVector*)VertBuffer->ptr();
	OriginalModel.VertexCount = VertBuffer->count();
	OriginalModel.VertexNormal = VertNormalBuffer->ptr();
	//NYI
	OriginalModel.TrianglesAdditionInfo = nullptr;
#endif
	//OPTIX INIT

	LevelIndixes = PrimeContext->createBufferDesc(RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBTrisIndexBuffer.ptr());
	LevelIndixes->setRange(0, CDBTrisIndexBuffer.count());

	LevelVertexes = PrimeContext->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBVertexesBuffer.ptr());
	LevelVertexes->setRange(0, CDBVertexesBuffer.count());

	LevelModel = PrimeContext->createModel();
	LevelModel->setTriangles(LevelIndixes, LevelVertexes);
	LevelModel->update(0);

	Progress(0.7f);

	{
		//###TEXTURES
		xr_vector <xrHardwareTexture> TextureDescription; TextureDescription.reserve(Textures.size());
		TexturesData.resize(Textures.size());
		ZeroMemory(TexturesData.data(), TexturesData.size() * sizeof(void*));
		u32 OverallTextures = Textures.size();
		for (u32 TextureID = 0; TextureID < OverallTextures; TextureID++)
		{
			b_texture& Tex = Textures[TextureID];
			//Manually alloc device ptr to hold texture data
			__int64 TextureMemSize = (Tex.dwHeight * Tex.dwWidth) * sizeof(u32);
			
			////Msg("H: %d, W: %d", Tex.dwHeight, Tex.dwWidth);
			u32* raw = static_cast<u32*>(*Tex.pSurface);

			if (raw != nullptr)
			{
				TexturesData[TextureID] = new DeviceBuffer < char >(TextureMemSize, GetBufferTypeByMode(mode), use_GPU_mem_tex);
				TexturesData[TextureID]->copyToBuffer((char*)raw, TextureMemSize);
			}

			//create tex description
			xrHardwareTexture TexDesc;
			TexDesc.Width = Tex.dwWidth;
			TexDesc.Height = Tex.dwHeight;
			TexDesc.IsHaveAlpha = !!Tex.bHasAlpha;
			if ( raw != nullptr)
			{
 				TexDesc.Pixels = (xrHardwarePixel*) TexturesData[TextureID]->ptr();
			}
			else
			{
				TexDesc.Pixels = nullptr;
			}

			TextureDescription.push_back(TexDesc);

			float CurrentStageProgress = float(TextureID) / float(OverallTextures);
			Progress(0.75f + (0.2f * CurrentStageProgress));
		}

		TextureBuffer = new DeviceBuffer<xrHardwareTexture>(Textures.size(), GetBufferTypeByMode(mode), use_GPU_mem_tex);
		TextureBuffer->copyToBuffer(TextureDescription.data(), Textures.size());
	}
	 

	Progress(0.95f);
	//###LIGHT INFO

	int RGBSize = Lightings.rgb.size();
	int SunSize = Lightings.sun.size();
	int HemiSize = Lightings.hemi.size();
	LightSizeInfo LightSize{ RGBSize, SunSize, HemiSize };
	LightSizeBuffer->copyToBuffer(&LightSize, 1);

	LightBuffer = new DeviceBuffer<R_Light>(RGBSize + SunSize + HemiSize, GetBufferTypeByMode(mode), use_GPU_mem);
	R_Light* LightDevicePtr = LightBuffer->ptr();
	size_t LightPtrOffset = 0;

	//	RGB
	LightBuffer->copyToBuffer(Lightings.rgb.data(), RGBSize, LightPtrOffset);
	LightPtrOffset += RGBSize;

	//	Sun
	LightBuffer->copyToBuffer(Lightings.sun.data(), SunSize, LightPtrOffset);
	LightPtrOffset += SunSize;

	//	Hemi
	LightBuffer->copyToBuffer(Lightings.hemi.data(), HemiSize, LightPtrOffset);

	//finally create a single master ptr, to access all data in one place
	GlobalData = new DeviceBuffer<xrHardwareLCGlobalData>(1, GetBufferTypeByMode(mode), use_GPU_mem);
	xrHardwareLCGlobalData StructGlobalData;

	StructGlobalData.LightData = LightBuffer->ptr();
	StructGlobalData.LightSize = LightSizeBuffer->ptr();
	StructGlobalData.Textures  = TextureBuffer->ptr();
	StructGlobalData.RaycastModel = OptimizedModel;

	GlobalData->copyToBuffer(&StructGlobalData, 1);
	Progress(1.0f);

}

void xrHardwareLight::PerformRaycast(xr_vector<RayRequest>& InRays, int flag, xr_vector<base_color_c>& OutHits)
{
	//We use all static light in our calculations for now
	if (InRays.empty())
	{
		//Msg("! PerformRaycast: Invoked without rays...");
		return;
	}

	cudaError_t DebugErr = cudaError_t::cudaSuccess;
	base_lighting& AllLights = inlc_global_data()->L_static();

	auto ZeroOutput = [](int RequestedRays, xr_vector<base_color_c>& TargetArray)
	{
		TargetArray.clear();
		TargetArray.reserve(RequestedRays);

		base_color_c ZeroColor;

		for (int i = 0; i < RequestedRays; ++i)
		{
			TargetArray.push_back(ZeroColor);
		}
	};

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
		//Msg("! PerformRaycast invoked, but no lights can be accepted");
		//Msg("All static lights RGB: %d Sun: %d Hemi: %d", AllLights.rgb.size(), AllLights.sun.size(), AllLights.hemi.size());
		return;
	}
 

	u64 MaxPotentialRays = MaxRaysPerPoint * InRays.size();
	
	Msg("MEM CPY START %u", InRays.size() * sizeof(RayRequest) / 1024 / 1024);

	DeviceBuffer < RayRequest > RayBuffer(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_1);
	DebugErr = cudaMemcpy(RayBuffer.ptr(), InRays.data(), InRays.size() * sizeof(RayRequest), cudaMemcpyHostToDevice);			  //
	CheckCudaErr(DebugErr);
	  
	MemoryUSE.start_1 = MaxPotentialRays / 1024 / 1024;
	Msg("MEM SET START %u", MaxPotentialRays / 1024 / 1024);

	DeviceBuffer<char> RayEnableStatusBuffer(MaxPotentialRays, GetBufferTypeByMode(mode), use_GPU_mem_stage_1);
	DebugErr = cudaMemset(RayEnableStatusBuffer.ptr(), 0, MaxPotentialRays);
	CheckCudaErr(DebugErr);

	Msg("StartOptimize out");
	CTimer t; t.Start();
	DebugErr = RunCheckRayOptimizeOut(GlobalData->ptr(), RayBuffer.ptr(), RayEnableStatusBuffer.ptr(), MaxPotentialRays, flag);
	CheckCudaErr(DebugErr);
	Msg("Optimize: %d", t.GetElapsed_ms());

	const char* StatusBuffer = RayEnableStatusBuffer.hostPtr();
	char* pItStatusBuffer = const_cast <char*> (StatusBuffer);


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

//	Msg("SizeCalculated: %d", AliveRaysIndexes.size());
  
	if (AliveRaysIndexes.empty())
	{
		//all rays are optimized
		ZeroOutput(InRays.size(), OutHits);
		return;
	}
	
	//create rays buffer and fill them through cuda
	Msg("MEM CPY 2 START %u", AliveRaysIndexes.size() * sizeof(u32) / 1024 / 1024);
	MemoryUSE.start_2 = AliveRaysIndexes.size() * sizeof(u32) / 1024 / 1024;
	DeviceBuffer <u32> DeviceAliveRaysIndexes(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2);
	DebugErr = cudaMemcpy(DeviceAliveRaysIndexes.ptr(), AliveRaysIndexes.data(), AliveRaysIndexes.size() * sizeof(u32), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);																				 
	
	Msg("MEM CPY 3 START %u", SurfacePoint2StartLoc.size() * sizeof(int) / 1024 / 1024);
	MemoryUSE.start_3 = AliveRaysIndexes.size() * sizeof(u32) / 1024 / 1024;

	DeviceBuffer <int> DeviceSurfacePoint2StartLoc(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2);
	DebugErr = cudaMemcpy(DeviceSurfacePoint2StartLoc.ptr(), SurfacePoint2StartLoc.data(), SurfacePoint2StartLoc.size() * sizeof(int), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);
	 
	Msg("MemSet OptimizedRaysVec: %u", AliveRaysIndexes.size() * sizeof(Ray) / 1024 / 1024);
	MemoryUSE.start_4 = AliveRaysIndexes.size() * sizeof(Ray) / 1024 / 1024;

	DeviceBuffer<Ray> OptimizedRaysVec(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2);  //use_GPU_mem
	DebugErr = RunGenerateRaysForTask(GlobalData->ptr(), RayBuffer.ptr(), OptimizedRaysVec.ptr(), DeviceAliveRaysIndexes.ptr(), AliveRaysIndexes.size(), flag);
	CheckCudaErr(DebugErr);
	
	Msg("MemSet OptimizedHitsVec: %u", AliveRaysIndexes.size() * sizeof(Hit) / 1024 / 1024);
	MemoryUSE.start_5 = AliveRaysIndexes.size() * sizeof(Hit) / 1024 / 1024;

	t.Start();
	DeviceBuffer <Hit> OptimizedHitsVec(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_2);
	DebugErr = cudaMemset(OptimizedHitsVec.ptr(), 0, OptimizedHitsVec.count() * sizeof(Hit));
	CheckCudaErr(DebugErr);
	 
	Msg("ProcessHITS: %d ms", t.GetElapsed_ms());


	optix::prime::BufferDesc OptmizedHitDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Hit::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedHitsVec.ptr());
	OptmizedHitDescBuffer->setRange(0, OptimizedHitsVec.count());

	optix::prime::BufferDesc OptimizedRaysDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Ray::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedRaysVec.ptr());
	OptimizedRaysDescBuffer->setRange(0, OptimizedRaysVec.count());

	optix::prime::Query LevelQuery = LevelModel->createQuery(RTP_QUERY_TYPE_CLOSEST);
	LevelQuery->setHits(OptmizedHitDescBuffer);
	LevelQuery->setRays(OptimizedRaysDescBuffer);

	//if "skip face" mode enabled - load special buffer for every requested surface point


	bool SkipFaceMode = !!(flag & LP_UseFaceDisable);
	DeviceBuffer <u64> FaceToSkip(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3);

	//go go round system!

	u32 AliveRays = AliveRaysIndexes.size();
	DeviceBuffer<float> RayEnergy(AliveRaysIndexes.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3);
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

	//OptimizedRaysVec.hostPtr();
	t.Start();
	while (AliveRays)
	{
		Msg("ALiveRays: %d", AliveRays);
		CTimer tt; tt.Start();
		LevelQuery->execute(0);
		//Msg("execute: %d", tt.GetElapsed_ms());
		AliveRays = 0;



		DebugErr = RunProcessHits(GlobalData->ptr(),
			OptimizedRaysVec.ptr(),
			OptimizedHitsVec.ptr(),
			RayEnergy.ptr(),
			RayEnableStatusBuffer.ptr(),
			DeviceAliveRaysIndexes.ptr(),
			AliveRaysIndexes.size(),
			IsFirstCall,
			flag,
			FacesToSkip);
		CheckCudaErr(DebugErr);
	
		const char* pIsRayAlive = RayEnableStatusBuffer.hostPtr();

		for (int Index = 0; Index < RayEnableStatusBuffer.count(); ++Index)
		{
			if (pIsRayAlive[Index]) ++AliveRays;
		}

		if (IsFirstCall) IsFirstCall = false;
		++Rounds;
	}

	Msg("LEVEL QUARY: %d ms", t.GetElapsed_ms());
  
	ZeroOutput(InRays.size(), OutHits);

 	MemoryUSE.start_6 = OutHits.size() * sizeof(base_color_c) / 1024 / 1024;
   
	DeviceBuffer <base_color_c> FinalColorBuffer(InRays.size(), GetBufferTypeByMode(mode), use_GPU_mem_stage_3);
	DebugErr = cudaMemcpy(FinalColorBuffer.ptr(), OutHits.data(), OutHits.size() * sizeof(base_color_c), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);
  
	DebugErr = RunFinalizeRays(
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
 
	//FinalColorBuffer.hostPtr();
	//copy directly back
	DebugErr = cudaMemcpy(OutHits.data(), FinalColorBuffer.ptr(), FinalColorBuffer.count() * sizeof(base_color_c), cudaMemcpyDeviceToHost);
	CheckCudaErr(DebugErr);
}

void xrHardwareLight::PerformAdaptiveHT()
{
	Progress(0.0f);
	cudaError_t DebugErr = cudaError_t::cudaSuccess;

	vecVertex& LevelVertexData = inlc_global_data()->g_vertices();


	int OverallVertexes = LevelVertexData.size();

 	xr_vector<RayRequest> AdaptiveHTRays;
	AdaptiveHTRays.reserve(LevelVertexData.size());

	for (int VertexIndex = 0; VertexIndex < OverallVertexes; ++VertexIndex)
	{
		Vertex* Vert = LevelVertexData[VertexIndex];
		RayRequest NewRequest{ Vert->P, Vert->N, nullptr };
		AdaptiveHTRays.push_back(NewRequest);
	}

	Progress(0.1f);

	xr_vector<base_color_c> tempColors;
	//PerformRaycast(AdaptiveHTRays, LP_dont_rgb + LP_dont_sun, tempColors);
	
	CTimer t; t.Start();
	PerformRaycast(AdaptiveHTRays, LP_dont_rgb + LP_dont_sun, tempColors);
	Msg("Elapsed: %.5f", t.GetElapsed_sec());

	
	Progress(0.9f);


	for (int VertexIndex = 0; VertexIndex < OverallVertexes; ++VertexIndex)
	{
		Vertex* Vert = LevelVertexData[VertexIndex];
		base_color_c& VertexColor = tempColors[VertexIndex];

		VertexColor.mul(0.5f);
		Vert->C._set(VertexColor);
	}
	
	 

//	//Msg("Size(%d)", AdaptiveHTRays.size());
	Progress(1.0f);
}

float xrHardwareLight::GetEnergyFromSelectedLight(xr_vector<int>& RGBLightIndexes, xr_vector<int>& SunLightIndexes, xr_vector<int>& HemiLightIndexes)
{

	DeviceBuffer <int> DeviceRGBLightIndexes(RGBLightIndexes.size(), GetBufferTypeByMode(mode));
	cudaMemcpy(DeviceRGBLightIndexes.ptr(), RGBLightIndexes.data(), sizeof(int) * RGBLightIndexes.size(), cudaMemcpyHostToDevice);

	DeviceBuffer <int> DeviceSunLightIndexes(SunLightIndexes.size(), GetBufferTypeByMode(mode));
	cudaMemcpy(DeviceSunLightIndexes.ptr(), SunLightIndexes.data(), sizeof(int) * SunLightIndexes.size(), cudaMemcpyHostToDevice);

	DeviceBuffer <int> DeviceHemiLightIndexes(HemiLightIndexes.size(), GetBufferTypeByMode(mode));
	cudaMemcpy(DeviceHemiLightIndexes.ptr(), HemiLightIndexes.data(), sizeof(int) * HemiLightIndexes.size(), cudaMemcpyHostToDevice);
	//count overall needed rays
	int OverallRaysCount = RGBLightIndexes.size() + SunLightIndexes.size() + HemiLightIndexes.size();

	//and this is the most vulnurable part of all cycle
	DeviceBuffer<Hit> HitsVec(OverallRaysCount, GetBufferTypeByMode(mode));
	optix::prime::BufferDesc HitDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Hit::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, HitsVec.ptr());
	HitDescBuffer->setRange(0, OverallRaysCount);

	DeviceBuffer<Ray> RaysVec(OverallRaysCount, GetBufferTypeByMode(mode));
	//and now, i summon my most handfull friend - CUDA, i choose YOU!

	optix::prime::BufferDesc RaysDescBuffer = PrimeContext->createBufferDesc((RTPbufferformat)Ray::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, RaysVec.ptr());
	RaysDescBuffer->setRange(0, OverallRaysCount);


	return 1.0f;
}

  
xr_vector<Fvector> xrHardwareLight::GetDebugPCHitData()
{
	return CPURayVertexData;
}

xr_vector<Fvector> xrHardwareLight::GetDebugGPUHitData()
{
	return GPURayVertexData;
}

void xrHardwareLight::GetLevelIndices(vecVertex& InLevelVertices, vecFace& InLevelFaces, xr_vector <PolyIndexes>& OutLevelIndices, xr_vector<HardwareVector>& OutLevelVertexes)
{
	//display Progress

	OutLevelVertexes.reserve(InLevelVertices.size());
	OutLevelIndices.reserve(InLevelFaces.size() * 3);
	xr_map <Vertex*, int> IndexMap;

	//from 0.1 to 0.3
	int OverallLevelIndexes = InLevelVertices.size();
	for (int VertexIndex = 0; VertexIndex < OverallLevelIndexes; ++VertexIndex)
	{
		Vertex* Vert = InLevelVertices[VertexIndex];

		OutLevelVertexes.emplace_back(Vert->P);
		IndexMap.insert(std::pair<Vertex*, int>(Vert, VertexIndex));

		float CurrentStageProgress = float(VertexIndex) / float(OverallLevelIndexes);
		Progress(0.1f + (0.2f * CurrentStageProgress));
	}

	//from 0.3 to 0.6
	int OverallFaceIndexes = InLevelFaces.size();

	for (int FaceIndex = 0; FaceIndex < OverallFaceIndexes; ++FaceIndex)
	{
		Face* FaceObj = InLevelFaces[FaceIndex];

		PolyIndexes FaceIndexes;

		auto IndexValue = IndexMap.find(FaceObj->v[0]);
		FaceIndexes.Index1 = IndexValue->second;

		IndexValue = IndexMap.find(FaceObj->v[1]);
		FaceIndexes.Index2 = IndexValue->second;

		IndexValue = IndexMap.find(FaceObj->v[2]);
		FaceIndexes.Index3 = IndexValue->second;

		OutLevelIndices.push_back(FaceIndexes);

		float CurrentStageProgress = float(FaceIndex) / float(OverallFaceIndexes);
		Progress(0.3f + (0.3f * CurrentStageProgress));
	}
}

void xrHardwareLight::CheckCudaError(cudaError_t ErrorCode)
{
	if (ErrorCode != cudaError::cudaSuccess)
	{
		//Msg(" [xrHardwareLight]: One of the call return non successful cuda error %d", ErrorCode);
	}
}

size_t xrHardwareLight::GetDeviceFreeMem()
{
	// Get amount of available video RAM
	switch (mode)
	{
		case xrHardwareLight::Mode::CUDA:
		{
			size_t TotalMem = 0;
			size_t FreeMem = 0;

			CheckCudaError(cudaMemGetInfo(&FreeMem, &TotalMem));
			return FreeMem;
		}
		default:
			return 0;
	}

}
  

const char* xrHardwareLight::ToString_Mode(Mode mode)
{
	switch (mode)
	{
		case xrHardwareLight::Mode::CUDA:
			return "CUDA";
		default:
			return "Unknown";
	}
}


size_t xrHardwareLight::GetMemoryRequiredForLoadLevel(CDB::MODEL* RaycastModel, base_lighting& Lightings, xr_vector<b_BuildTexture>& Textures)
{
	size_t VertexDataSize = RaycastModel->get_verts_count() * sizeof(HardwareVector);
	size_t TrisIndexSize = RaycastModel->get_tris_count() * sizeof(PolyIndexes);
	size_t TrisAdditionalDataSize = RaycastModel->get_tris_count() * sizeof(TrisAdditionInfo);

	size_t OptixMeshDataOverhead = VertexDataSize + TrisIndexSize;

	size_t TextureMemorySize = 0;
	for (const b_BuildTexture& Texture : Textures)
	{
		size_t TextureSize = (Texture.dwHeight * Texture.dwWidth) * sizeof(u32);
		TextureSize += sizeof(xrHardwareTexture);
		TextureMemorySize += TextureSize;
	}

	size_t LightingInfoSize = (Lightings.rgb.size() + Lightings.sun.size() + Lightings.hemi.size()) * sizeof(R_Light);
	size_t TotalMemorySize = VertexDataSize + TrisIndexSize + TrisAdditionalDataSize + OptixMeshDataOverhead + TextureMemorySize + LightingInfoSize;

	//Msg(" [xrHardwareLight]: Vertex data size: %zu MB, Tris index size: %zu MB", VertexDataSize / 1024 / 1024, TrisIndexSize / 1024 / 1024);
	//Msg(" [xrHardwareLight]: Tris Additional Data: %zu MB", TrisAdditionalDataSize / 1024 / 1024);
	//Msg(" [xrHardwareLight]: OptiX overhead: %zu MB", OptixMeshDataOverhead / 1024 / 1024);
	//Msg(" [xrHardwareLight]: Overall texture memory: %zu MB", TextureMemorySize / 1024 / 1024);
	//Msg(" [xrHardwareLight]: Lighting: %zu MB", LightingInfoSize / 1024 / 1024);
	//Msg(" [xrHardwareLight]: TOTAL: %zu MB", TotalMemorySize / 1024 / 1024);

	return TotalMemorySize;
}


bool xrHardwareLight::_IsEnabled;
