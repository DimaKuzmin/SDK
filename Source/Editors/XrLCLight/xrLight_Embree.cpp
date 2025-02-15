#include "stdafx.h"
 
#include "xrLight_Embree.h"
#include "../../xrcdb/xrcdb.h"

#include "xrLC_GlobalData.h"
#include "xrface.h"
#include "xrdeflector.h"
#include "light_point.h"
#include "R_light.h"

//Intel Code Start
#include "EmbreeDataStorage.h"
#include <atomic>

// Важные параметры
// INTIALIZE GEOMETRY, SCENE QUALITY TYPE
// Инициализация Основных Фишек Embree

// #define USE_TRANSPARENT_GEOM

#include "BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

// INTEL DATA STRUCTURE
int LastGeometryID = RTC_INVALID_GEOMETRY_ID;
 
RTCDevice device;
RTCScene IntelScene = 0;

RTCGeometry IntelGeometryNormal = 0;
 
/** NORMAL GEOM **/
Embree::VertexEmbree* verticesNormal = 0;
Embree::TriEmbree* trianglesNormal = 0;
xr_vector<void*> TriNormal_Dummys;
 
// Сильно ускоряет Но не нужно сильно завышать вообще 0.01f желаетельно 
// Влияет на яркость на выходе (если близко к 0 будет занулятся)
// можно и 0.10f Было раньше так
float EmbreeEnergyMAX = 0.01f;

struct RayQueryContext
{
	RTCRayQueryContext context;
	Fvector B;

	Face* skip = 0;
	R_Light* Light = 0;
	float energy = 1.0f;
	u32 Hits = 0;
};

ICF bool CalculateEnergy(base_Face* F, Fvector& B, float& energy, float u, float v)
{		
	// Перемещаем начало луча немного дальше пересечения
	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];

	// barycentric coords
	// note: W,U,V order
	B.set(1.0f - u - v, u, v);

	//// calc UV
	Fvector2* cuv = F->getTC0();
	Fvector2	uv;
	uv.x = cuv[0].x * B.x + cuv[1].x * B.y + cuv[2].x * B.z;
	uv.y = cuv[0].y * B.x + cuv[1].y * B.y + cuv[2].y * B.z;
	int U = iFloor(uv.x * float(T.dwWidth) + .5f);
	int V = iFloor(uv.y * float(T.dwHeight) + .5f);
	U %= T.dwWidth;		if (U < 0) U += T.dwWidth;
	V %= T.dwHeight;	if (V < 0) V += T.dwHeight;


	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel = raw[V * T.dwWidth + U];
	u32 pixel_a = color_get_A(pixel);
	float opac = 1.f - _sqr(float(pixel_a) / 255.f);

	// Дополнение Контекста
	energy *= opac;
	if (energy < EmbreeEnergyMAX)
		return false;

	return true;
}

ICF void FilterRaytrace(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
 
	// Собрать все
  	base_Face* F = (base_Face*) TriNormal_Dummys[hit->primID];
	if (F == ctxt->skip || !F)
	{
		args->valid[0] = 0;  return;
	}
 	if (F->flags.bOpaque)
	{
		ctxt->energy = 0;  return;
	}
	 
 	if (!CalculateEnergy(F, ctxt->B, ctxt->energy, hit->u, hit->v))
	{
		// При нахождении любого хита сразу все попали в непрозрачный Face.
 		ctxt->energy = 0; return;
	} 
	args->valid[0] = 0; // Задаем чтобы продолжил поиск
} 

float RaytraceEmbreeProcess( R_Light& L, Fvector& P, Fvector& N, float range, void* skip)
{
	// Структура для RayTracing
	RayQueryContext data_hits;
	data_hits.Light = &L;
	data_hits.skip = (Face*) skip;
	data_hits.energy = 1.0f;
	data_hits.Hits = 0;
	 	
	/// Непрозрачные чекаем
 
	RTCRay ray;
	Embree::SetRay1(ray, P, N, 0.01f, range);
 
	RTCOccludedArguments args;
	rtcInitOccludedArguments(&args);

	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	// SET CONTEXT
	data_hits.context = context;
	args.context = &data_hits.context;
	args.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;

	rtcOccluded1(IntelScene, &ray, &args);
	return data_hits.energy;
}

// REGISTER(LOAD) GEOMETRY
size_t Predcalculated = 0;



void InitializeGeometryAttach_new(bool isTransp)
{
	Predcalculated = 0;

	// Precalucalate
 	{
		std::atomic<size_t> TrianglesSize = 0;
		Embree::GetGlobalData(isTransp, TRUE, TrianglesSize, nullptr, nullptr, nullptr); // Без буферов !!!
		Predcalculated = TrianglesSize;
 		clMsg("[Embree] Faces [Precalc]: %u ", Predcalculated);
	}

	// Get Buffers By Type Geometry
	xr_vector<void*>& dummy				= TriNormal_Dummys;
	RTCGeometry& RtcGeometry			= IntelGeometryNormal;
	Embree::VertexEmbree* vertex_embree = verticesNormal;
	Embree::TriEmbree* tri_embree		= trianglesNormal;

	// RtcIntilize Geoms
	RtcGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(RtcGeometry, (RTCBuildQuality)build_args->EmbreeGeomType);

	rtcSetGeometryOccludedFilterFunction(RtcGeometry, &FilterRaytrace);
	rtcSetGeometryIntersectFilterFunction(RtcGeometry, &FilterRaytrace);

	// GET TRIANGLE (COLLECTORs Data) 
	 
 	vertex_embree = (Embree::VertexEmbree*)rtcSetNewGeometryBuffer(RtcGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Embree::VertexEmbree), Predcalculated * 3);
 	tri_embree = (Embree::TriEmbree*)rtcSetNewGeometryBuffer(RtcGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Embree::TriEmbree), Predcalculated);
 
	// FIX
	dummy.clear();
	dummy.resize(Predcalculated);
 
	// Set Buffer Data
 	std::atomic<size_t> TrianglesSize = 0;
	GetGlobalData(isTransp, FALSE, TrianglesSize, vertex_embree, tri_embree, &dummy); // Указать буферы !!!
 	rtcCommitGeometry(RtcGeometry);
  
	LastGeometryID = rtcAttachGeometry(IntelScene, RtcGeometry);
	clMsg("[Intel Embree] Attached Geometry: IntelGeometry(%s) By ID: %d, Traingles: %u",
		"Normal",
		LastGeometryID,
		Predcalculated);
}

void RemoveGeoms()
{
	size_t free, relocated, used;
	vminfo(&free, &relocated, &used);
	clMsg("[Embree][memory][PRE] : RemoveGeoms mem: %u", used / 1024 / 1024);

	if (LastGeometryID != RTC_INVALID_GEOMETRY_ID)
	{
		
		rtcDetachGeometry(IntelScene, LastGeometryID);
		rtcReleaseGeometry(IntelGeometryNormal);
		
		size_t free, relocated, used;
		vminfo(&free, &relocated, &used);
		clMsg("[Embree][memory] : Release Geom: %u, mem: %u", IntelGeometryNormal, used / 1024 / 1024);

		verticesNormal = 0;
		trianglesNormal = 0;
		TriNormal_Dummys.clear();

		LastGeometryID = RTC_INVALID_GEOMETRY_ID;
	}
}

XRLC_LIGHT_API void IntelEmbereDetachRelease()
{
	if (IntelScene != nullptr)
	{
 		IntelEmbereUNLOAD();
	}
}

XRLC_LIGHT_API void IntelEmbereLOAD()
{
	if (IntelScene != nullptr)
	{
		RemoveGeoms();
	}
	else
	{
		bool avx_test = build_args->use_avx;
		bool sse = build_args->use_sse;

		const char* config = "";
		if (avx_test)
			config = "threads=16,isa=avx2";
		else if (sse)
			config = "threads=16,isa=sse4.2";
		else
			config = "threads=16,isa=sse2";

		device = rtcNewDevice(config);
		rtcSetDeviceErrorFunction(device, Embree::errorFunction, NULL);


		string128 phase;
		sprintf(phase, "Intilized Intel Embree %s - %s", RTC_VERSION_STRING, avx_test ? "avx" : sse ? "sse" : "default");
		Status(phase);
		Embree::IntelEmbreeSettings(device, avx_test, sse);

		// Создание сцены и добавление геометрии
		// Scene
		IntelScene = rtcNewScene(device);

		RTCSceneFlags scene_flags;

		if (build_args->useRobust)
			scene_flags = RTC_SCENE_FLAG_ROBUST;
		else
			scene_flags = RTC_SCENE_FLAG_COMPACT;

		rtcSetSceneFlags(IntelScene, scene_flags);
	}

	size_t free, used, reserved;
	vminfo(&free, &reserved, &used);
	clMsg("[Embree][memory][SceneStart] Memory Used: %u mb", used / 1024 / 1024);

	InitializeGeometryAttach_new(false); /// GeomID == 0
	 
	rtcCommitScene(IntelScene);

	Memory.mem_compact();
	 
	vminfo(&free, &reserved, &used);
	clMsg("[Embree][memory][SceneCommit] Memory Used: %u mb", used / 1024 / 1024);
} 

void IntelEmbereUNLOAD()
{
	RemoveGeoms();
	rtcReleaseScene(IntelScene);
	
	size_t free, relocated, used;
	vminfo(&free, &relocated, &used);
	clMsg("[Embree][memory] : Release Scene: mem: %u", used / 1024 / 1024);

	rtcReleaseDevice(device);

	vminfo(&free, &relocated, &used);
	clMsg("[Embree][memory] : Release Device: mem: %u", used / 1024 / 1024);

	IntelScene = 0;
	device = 0;
}
