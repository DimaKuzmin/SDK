#include "stdafx.h"
 
#include "EmbreeRayTrace.h"
#include "../xrCDB/xrCDB.h"

#include "xrLC_GlobalData.h"
#include "xrFace.h"
#include "xrDeflector.h"
#include "light_point.h"
#include "R_light.h"
 
#include "xrMU_Model.h"
#include "xrMU_Model_Reference.h"
 
// Важные параметры
// INTIALIZE GEOMETRY, SCENE QUALITY TYPE
// Инициализация Основных Фишек Embree

// INTEL DATA STRUCTURE
RTCSceneFlags scene_flags = RTC_SCENE_FLAG_NONE;
RTCBuildQuality scene_quality = RTC_BUILD_QUALITY_LOW;

RTCDevice device = 0;
RTCScene IntelScene = 0;

RTCGeometry IntelGeometryNormal = 0;
RTCGeometry IntelGeometryMuModels = 0;

RTCGeometry IntelGeometryTransp = 0;
RTCGeometry IntelGeometryMuModelsTransp = 0;

XRLC_LIGHT_API EmbreeData EmbreeMain;

// Сильно ускоряет Но не нужно сильно завышать вообще 0.01f желаетельно 
// Влияет на яркость на выходе (если близко к 0 будет занулятся)
// можно и 0.10f Было раньше так
float EmbreeEnergyMAX = 0.035f;

struct RayQueryContext
{
	RTCRayQueryContext context;
	Fvector B;
	
	Face* face;
 	Face* skip	  = 0;
	R_Light* Light = 0;
	float energy = 1.0f;
	u32 Hits = 0;

	// Texture Coord U
	int tU;
	// Texture Coord V
	int tV;

	// Barycentric UV
	Fvector2 uv;
};
 
static float opacityLUT[256] = { 0 };
static float decayLUT[32] = {0};

void InitOpacityLUT() 
{
	for (int i = 0; i < 256; ++i) 
	{
		float a = i / 255.f;
		opacityLUT[i] = 1.f - _sqr(a);
	}

	for (int i = 0; i < 32; ++i)
		decayLUT[i] = powf(0.95f, i);
}

// Сделать потом переключалку
bool CalculateEnergy(RayQueryContext*ctxt, RTCHit* hit)
{
	// Перемещаем начало луча немного дальше пересечения
	b_material& M = inlc_global_data()->materials()[ctxt->face->dwMaterial];
	b_texture&  T = inlc_global_data()->textures()[M.surfidx];

	// barycentric coords
	// note: W,U,V order
	ctxt->B.set(1.0f - hit->u - hit->v, hit->u, hit->v);

	//// calc UV
	Fvector2*	cuv = ctxt->face->getTC0();
  	ctxt->uv.x = cuv[0].x * ctxt->B.x + cuv[1].x * ctxt->B.y + cuv[2].x * ctxt->B.z;
	ctxt->uv.y = cuv[0].y * ctxt->B.x + cuv[1].y * ctxt->B.y + cuv[2].y * ctxt->B.z; 
	
	// Без floor быстрее и работает хорошо
	ctxt->tU = int(ctxt->uv.x * T.dwWidth + 0.5f);
	ctxt->tV = int(ctxt->uv.y * T.dwHeight + 0.5f);
	ctxt->tU = (ctxt->tU % T.dwWidth + T.dwWidth) % T.dwWidth;
	ctxt->tV = (ctxt->tV % T.dwHeight + T.dwHeight) % T.dwHeight;
 
	// Прозрачность
	u32* surface	 = static_cast<u32*>(*T.pSurface);
	u32 opacity		 = color_get_A(surface[ctxt->tV * T.dwWidth + ctxt->tU]);
	// Используем заранее посчитаные данные 
   	// Дополнение Контекста

	float opac = opacityLUT[opacity];
	ctxt->energy *= opac;
	ctxt->Hits++;

 	// Отымаем енергию чтобы быстрее выйти
	if (ctxt->Hits > 1)  
 		ctxt->energy *= decayLUT[ctxt->Hits];
 
	return ctxt->energy > EmbreeEnergyMAX;
}

void FilterRaytraceTransparent(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt	= (RayQueryContext*)args->context;
	RTCHit* hit				= (RTCHit*)args->hit;
 
	// Собрать все
	Face* F = hit->geomID == 2 ? EmbreeMain.static_geom_transp.dummy[hit->primID] : EmbreeMain.murefs_geom_transp.dummy[hit->primID];
 	
	ctxt->face = F;
  	if (F != ctxt->skip && !CalculateEnergy(ctxt, hit))
 		ctxt->energy = 0; // Отсеили 
 	else 
		args->valid[0] = 0;	// Продолжаем
}


void FilterRayTraceOpaque(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;

	Face* F = hit->geomID == 0 ? EmbreeMain.static_geom.dummy[hit->primID] : EmbreeMain.murefs_geom.dummy[hit->primID];
	if (F == ctxt->skip)
	{
		args->valid[0] = 0;
		return;
	}
	ctxt->energy = 0; // Отсеили 
}


float EmbreeData::RaytraceEmbreeProcess(R_Light& L, Fvector& P, Fvector& N, float range, void* skip)
{
 	// Структура для RayTracing
	RayQueryContext data_hits;
	data_hits.Light = &L;
	data_hits.skip = (Face*)skip;
	data_hits.energy = 1.0f;
	data_hits.Hits = 0;

	RTCRay ray;
	SetRay1(ray, P, N, 0.001f, range);

	RTCOccludedArguments args;
	rtcInitOccludedArguments(&args);

	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	// SET CONTEXT
	data_hits.context = context;
	args.context = &data_hits.context;
	rtcOccluded1(IntelScene, &ray, &args);

	return data_hits.energy;
}

// LOADING GEOMETRY

size_t GetMemory()
{
	size_t used, free, reserved;
	vminfo(&free, &reserved, &used);
	return used;
}

void LoadGeomBuffer(RTCGeometry& geom, RTCBuildQuality& quality, bool FilterTransp, TriangleContainer& geom_buffer)
{
	geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(geom, quality);

	if (FilterTransp)
		rtcSetGeometryOccludedFilterFunction(geom, &FilterRaytraceTransparent);
	else
		rtcSetGeometryOccludedFilterFunction(geom, &FilterRayTraceOpaque);

	rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, geom_buffer.vertex().data(), 0, sizeof(VertexEmbree), geom_buffer.vertex().size());
	rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, geom_buffer.faces().data(), 0, sizeof(TriEmbree), geom_buffer.faces().size());
	geom_buffer.hashTable.clear();

	rtcCommitGeometry(geom);
} 

void EmbreeData::InitializeGeometry(size_t& geom_static_mem, size_t& geom_murefs_mem, bool useMU)
{
	useMU = true;

	// Конструктор модели
	EmbreeData::GetGlobalData(geom_static_mem, geom_murefs_mem, useMU);

	LoadGeomBuffer(IntelGeometryNormal, scene_quality, false, static_geom);
	LoadGeomBuffer(IntelGeometryTransp, scene_quality, true, static_geom_transp);
  
	LoadGeomBuffer(IntelGeometryMuModels, scene_quality, false, murefs_geom);
	LoadGeomBuffer(IntelGeometryMuModelsTransp, scene_quality, true, murefs_geom_transp);

}

size_t EmbreeData::AttachGeometrys(bool addMU)
{
	RemoveGeometry(false);

	IntelScene = rtcNewScene(device);
	rtcSetSceneFlags(IntelScene, scene_flags);

	isAttached = true;
	rtcAttachGeometryByID(IntelScene, IntelGeometryNormal, 0);
	rtcAttachGeometryByID(IntelScene, IntelGeometryTransp, 2);

	if (murefs_geom.faces().size() > 0 && addMU)
	{
		rtcAttachGeometryByID(IntelScene, IntelGeometryMuModels, 1);
 		rtcAttachGeometryByID(IntelScene, IntelGeometryMuModelsTransp, 3);
	} 

	size_t start = GetMemory();
	rtcCommitScene(IntelScene);
	BVH_size = GetMemory() - start;

	Msg("Static MODELS Transp : %u, Opacue: %u", static_geom_transp.faces_v.size(), static_geom.faces_v.size());
	Msg("MU MODELS Transp : %u, Opacue: %u", murefs_geom_transp.faces_v.size(), murefs_geom.faces_v.size());

	return (GetMemory() - start);
}

void EmbreeData::RemoveGeometry(bool isDealloc)
{
	if (isDealloc)
	{
		rtcReleaseScene(IntelScene);
		static_geom.ClearAll();
		static_geom_transp.ClearAll();
		murefs_geom.ClearAll();
		murefs_geom_transp.ClearAll();

		BVH_size = 0;
		Static_size = 0;
		MU_size = 0;
	}
	else
	{
		rtcReleaseScene(IntelScene);
		BVH_size = 0;
	}

	IntelScene = 0;
}

void errors_embree(void* userPtr, enum RTCError code, const char* str)
{
	R_ASSERT2(false, str);
}

void EmbreeData::IntializeDevice()
{
	bool avx_test = true; 
	bool sse	  = true; 

	InitOpacityLUT();

	const char* config = "";
	if (avx_test)
		config = "threads=16,isa=avx2,verbose=0";
	else if (sse)
		config = "threads=16,isa=sse4.2,verbose=0";
	else
		config = "threads=16,isa=sse2,verbose=0";

	device = rtcNewDevice(config);

	rtcSetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED, 0);
	rtcSetDeviceErrorFunction(device, &errors_embree, NULL);

	string128 state;
	sprintf(state, "- Intilized Intel Embree %s - %s", RTC_VERSION_STRING, avx_test ? "avx" : sse ? "sse" : "default");
	Status(state);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED", device, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED", device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED", device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_TASKING_SYSTEM", device, RTC_DEVICE_PROPERTY_TASKING_SYSTEM);
}

void EmbreeData::IntelEmbereLOAD(bool useMU)
{
	if (!isInitialized)
	{
		IntializeDevice();
		isInitialized = true;
	}

	Msg("- Intel Embree Loading| Memory: %u mb", u32(GetMemory() / 1024 / 1024));
	if (gCompilerMode.EmbreeBVHCompact)
		scene_flags = scene_flags | RTC_SCENE_FLAG_COMPACT;
	if (gCompilerMode.EmbreeBVHRobust)
		scene_flags = scene_flags | RTC_SCENE_FLAG_ROBUST;

	IntelScene = rtcNewScene(device);
	rtcSetSceneFlags(IntelScene, scene_flags);

	// LOADING NORMAL GEOM
	size_t geom_memory, refs_memory;
	InitializeGeometry(geom_memory, refs_memory, useMU);

	size_t BVH = AttachGeometrys(true);
	AditionalData("ST: %umb | MU: %umb | BVH: %u mb", geom_memory / 1024 / 1024, refs_memory / 1024 / 1024, BVH / 1024 / 1024);
}

void EmbreeData::IntelEmbereUNLOAD()
{
	Msg("* Intel Embree Releasing Start| Memory: %u mb", u32(GetMemory() / 1024 / 1024));
	RemoveGeometry(true);
	Msg("* Intel Embree Releasing End| Memory: %u mb", u32(GetMemory() / 1024 / 1024));
}
