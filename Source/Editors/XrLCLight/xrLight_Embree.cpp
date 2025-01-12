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

// INTEL DATA STRUCTURE
 int LastGeometryID = RTC_INVALID_GEOMETRY_ID;
 
RTCDevice device;
RTCScene IntelScene = 0;

RTCGeometry IntelGeometryNormal = 0;

/** NORMAL GEOM **/
Embree::VertexEmbree* verticesNormal = 0;
Embree::TriEmbree* trianglesNormal = 0;
xr_vector<void*> TriNormal_Dummys;

// Качество сцены желательно REFIT юзать на более низких может криво отработать 
auto geom_type = RTCBuildQuality::RTC_BUILD_QUALITY_LOW;

// 20% Гдето задержка на ROBUST
auto scene_flags = RTCSceneFlags::RTC_SCENE_FLAG_NONE; 

// RTCSceneFlags::RTC_SCENE_FLAG_ROBUST; 

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

// Сделать потом переключалку

ICF void* GetGeomBuff(int GeomID, int Prim)
{
	if (GeomID == 0 && Prim != RTC_INVALID_GEOMETRY_ID)
		return TriNormal_Dummys[Prim];
		
	return nullptr;
}

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

extern XRLC_LIGHT_API int StageMAXHits = 32;

ICF void FilterRaytrace(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;	 
	
	// Собрать все
  	base_Face* F = (base_Face*)GetGeomBuff(hit->geomID, hit->primID);

	if (F->flags.bOpaque)
	{
		ctxt->energy = 0;
		return;
	}

	if (F == ctxt->skip || !F)
	{
		args->valid[0] = 0;
		return;
	}
	 
 	if (!CalculateEnergy(F, ctxt->B, ctxt->energy, hit->u, hit->v))
	{
		// При нахождении любого хита сразу все попали в непрозрачный Face.
 		ctxt->energy = 0;
		ctxt->Hits += 1;
		return;
	}

 	// ctxt->Hits += 1;
 	// if (ctxt->Hits > StageMAXHits)
	// 	return;

	args->valid[0] = 0; // Задаем чтобы продолжил поиск
}
 
// INITIALIZE CONTEXT
thread_local RTCRayQueryContext context;

 

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
	Embree::SetRay1(ray, P, N, range);
 
	RTCOccludedArguments args;
	rtcInitOccludedArguments(&args);
	rtcInitRayQueryContext(&context);

	// SET CONTEXT
	data_hits.context = context;
	args.context = &data_hits.context;

	rtcOccluded1(IntelScene, &ray, &args);
	return data_hits.energy;
}
 
void InitializeGeometryAttach_CDB(Fvector* CDB_verts, CDB::TRI* CDB_tris, u32 TS_Size)
{
	// NORMAL GEOM
	IntelGeometryNormal = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometryNormal, geom_type);

	rtcSetGeometryOccludedFilterFunction(IntelGeometryNormal, &FilterRaytrace);
	rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterRaytrace);

 	xr_vector<CDB::TRI*> Opacue;

 	for (auto i = 0; i < TS_Size; i++)
	{
		Opacue.push_back(&CDB_tris[i]);
	}
 
 	verticesNormal = (Embree::VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Embree::VertexEmbree), Opacue.size() * 3);
	trianglesNormal = (Embree::TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Embree::TriEmbree), Opacue.size());

	// FIX
	TriNormal_Dummys.clear(); TriNormal_Dummys.reserve(Opacue.size());

	size_t VertexIndexer = 0;
 	for (auto i = 0; i < Opacue.size(); i++)
	{
		trianglesNormal[i].SetVertexes(*Opacue[i], CDB_verts, verticesNormal, VertexIndexer);
		TriNormal_Dummys[i] = (Opacue[i]->pointer);
	}
 	rtcCommitGeometry(IntelGeometryNormal);
	LastGeometryID = rtcAttachGeometry(IntelScene, IntelGeometryNormal);
	rtcCommitScene(IntelScene);

	clMsg("Intel Initialized: Triangles: %d", Opacue.size());
	clMsg("[Intel Embree] Attached Geometry: IntelGeometry(Normal) By ID: %d", LastGeometryID);

	Opacue.clear();
}

void IntelEmbereLOAD(CDB::CollectorPacked& packed_cb)
{
	if (IntelScene != nullptr && LastGeometryID != RTC_INVALID_GEOMETRY_ID)
	{
		if (IntelGeometryNormal != nullptr)
		{
			rtcDetachGeometry(IntelScene, LastGeometryID);
			rtcReleaseGeometry(IntelGeometryNormal);

			verticesNormal = 0;
			trianglesNormal = 0;
			TriNormal_Dummys.clear();
			LastGeometryID = RTC_INVALID_GEOMETRY_ID;
		}
	}
	else
	{
		bool avx_test	= true;
		bool sse		= false;

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
 		rtcSetSceneFlags(IntelScene, scene_flags);
  	}

	InitializeGeometryAttach_CDB(packed_cb.getV(), packed_cb.getT(), packed_cb.getTS());
}

void IntelEmbereUNLOAD()
{
	if (LastGeometryID != RTC_INVALID_GEOMETRY_ID)
	{
		rtcDetachGeometry(IntelScene, LastGeometryID);
		rtcReleaseGeometry(IntelGeometryNormal);
	}
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}
