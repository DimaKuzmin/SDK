#include "stdafx.h"
#include "../../xrcdb/xrcdb.h"

//#include "xrLight_ImplicitDeflector.h"
//#include "xrlight_implicit.h"
//#include "xrlight_implicitcalcglobs.h"

#include "xrLC_GlobalData.h"
#include "xrface.h"
#include "xrdeflector.h"
#include "light_point.h"
#include "cl_intersect.h"
#include "R_light.h"

//Intel Code Start

#include "EmbreeDataStorage.h"
#include <atomic>


#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

#pragma comment(lib, "embree3.lib")
#pragma comment(lib, "tbb.lib")
#include "embree3/rtcore.h"


RTCScene IntelScene;
RTCScene IntelSceneTransparent;

RTCGeometry IntelGeometryNormal;
RTCGeometry IntelGeometryTransparent;


struct VertexEmbree
{
	float x, y, z;

	void Set(Fvector& vertex)
	{
		x = vertex.x;
		y = vertex.y;
		z = vertex.z;
	}
	void Get(Fvector& vertex)
	{
		vertex.x = x;
		vertex.y = y;
		vertex.z = z;
	}
};

struct TriEmbree
{
	uint32_t point1, point2, point3;
	void SetVertexes(Fvector* verts, VertexEmbree* emb_verts, size_t& last_index)
	{
		point1 = last_index;
		point2 = last_index + 1;
		point3 = last_index + 2;

		emb_verts[last_index].Set(verts[point1]);
		emb_verts[last_index + 1].Set(verts[point2]);
		emb_verts[last_index + 2].Set(verts[point3]);

		last_index += 3;
	}

	void GetVertex(VertexEmbree* emb_verts, Fvector& v1, Fvector& v2, Fvector& v3)
	{
  		emb_verts[point1].Get(v1);
		emb_verts[point2].Get(v2);
		emb_verts[point3].Get(v3);
	}
};


VertexEmbree* verticesNormal = 0;
TriEmbree* trianglesNormal = 0;
u32 SizeTriangleNormal = 0;
size_t SizeVertexNormal = 0;

VertexEmbree* verticesTransparent = 0;
TriEmbree* trianglesTransparent = 0;
u32 SizeTriangleTransparent = 0;
size_t SizeVertexTransparent = 0;

xr_vector<void*> TriNormal_Dummys;
xr_vector<void*> TriTransparent_Dummys;

RTCDevice device;

// ВАЖНЫЙ ПАРАМЕТР TNEAR Для пересечения с водой
int TNearParram = 0.2f;
 
void SetRay1(RayOptimizedCPU* ray, RTCRay& rayhit)
{
	Fvector posNew = ray->pos.mad(ray->dir, 0.1f);
	rayhit.dir_x = ray->dir.x;
	rayhit.dir_y = ray->dir.y;
	rayhit.dir_z = ray->dir.z;

	rayhit.org_x = posNew.x;
	rayhit.org_y = posNew.y;
	rayhit.org_z = posNew.z;

	rayhit.tnear = ray->tmin;
	rayhit.tfar = ray->tmax;

	rayhit.mask = (unsigned int)(-1);
	rayhit.flags = 0;
}
 
void SetRay1(RayOptimizedCPU* ray, RTCRayHit& rayhit)
{
	Fvector posNew = ray->pos.mad(ray->dir, 0.1f);
	rayhit.ray.dir_x = ray->dir.x;
	rayhit.ray.dir_y = ray->dir.y;
	rayhit.ray.dir_z = ray->dir.z;

	rayhit.ray.org_x = posNew.x;
	rayhit.ray.org_y = posNew.y;
	rayhit.ray.org_z = posNew.z;

	rayhit.ray.tnear = ray->tmin;
	rayhit.ray.tfar = ray->tmax;

	rayhit.ray.mask = (unsigned int)(-1);
	rayhit.ray.flags = 0;

	rayhit.hit.Ng_x = 0;
	rayhit.hit.Ng_y = 0;
	rayhit.hit.Ng_z = 0;

	rayhit.hit.u = 0;
	rayhit.hit.v = 0;

	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
}
 
void SetRay1Invert(RayOptimizedCPU* ray, RTCRay& rayhit)
{
	rayhit.dir_x = -ray->dir.x;
	rayhit.dir_y = -ray->dir.y;
	rayhit.dir_z = -ray->dir.z;

	Fvector pos = ray->pos.mad(ray->dir, ray->tmax);

	rayhit.org_x = pos.x;
	rayhit.org_y = pos.y;
	rayhit.org_z = pos.z;

	rayhit.tnear = ray->tmin;
	rayhit.tfar = ray->tmax;

	rayhit.mask = (unsigned int)(-1);
	rayhit.flags = 0;
}
 
void SetRay1Invert(RayOptimizedCPU* ray, RTCRayHit& rayhit)
{
	rayhit.ray.dir_x = -ray->dir.x;
	rayhit.ray.dir_y = -ray->dir.y;
	rayhit.ray.dir_z = -ray->dir.z;

	Fvector pos = ray->pos.mad(ray->dir, ray->tmax);

	rayhit.ray.org_x = pos.x;
	rayhit.ray.org_y = pos.y;
	rayhit.ray.org_z = pos.z;


	rayhit.ray.tnear = ray->tmin;
	rayhit.ray.tfar = ray->tmax;

	rayhit.ray.mask = (unsigned int)(-1);
	rayhit.ray.flags = 0;

	rayhit.hit.Ng_x = 0;
	rayhit.hit.Ng_y = 0;
	rayhit.hit.Ng_z = 0;

	rayhit.hit.u = 0;
	rayhit.hit.v = 0;

	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
}



// Сделать потом переключалку
 

struct RayQueryContext
{
	RTCIntersectContext context;
	Fvector B;
	Fvector StartPos;

	// CDB::MODEL* model = 0; 
	Face* skip = 0;
	R_Light* Light = 0;

	float energy = 1.0f;
	int hits = 0;
	bool Transparent = false;

	float Tfar = 0;
	int LastPremitive = 0;
	xr_vector<shared_str> dump_msg;
};

 
#define USE_OCCLUSION

// valid [0] = -1 (ПРИНЯТЬ ХИТ)
// valid [0] = 0 (ИГНОРИРОВАТЬ)

#ifdef USE_OCCLUSION
void FilterOcclusion(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCRay* ray = (RTCRay*)args->ray;
	RTCHit* hit = (RTCHit*)args->hit;
	
	args->valid[0] = 0;

	if (hit->primID == RTC_INVALID_GEOMETRY_ID || hit->primID == ctxt->LastPremitive)
		return;
	
	ctxt->hits++;
	ctxt->LastPremitive = hit->primID;

	ray->org_x = ray->org_x + (ray->dir_x * ray->tfar);
	ray->org_y = ray->org_y + (ray->dir_y * ray->tfar);
	ray->org_z = ray->org_z + (ray->dir_z * ray->tfar);

	ray->tnear = TNearParram;

	// Перемещаем начало луча немного дальше пересечения
 
	base_Face* F = 0;
 	if (hit->geomID == 1)
	{
		F = (base_Face*)(TriTransparent_Dummys[hit->primID]);
	}
	else
	{
		F = (base_Face*)(TriNormal_Dummys[hit->primID]);
	}
 	 
	if (F == nullptr || F == ctxt->skip)
  		return;
	
	// Access to texture
	if (F->flags.bOpaque ) // && ray->tfar > 3
	{
		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];
		
		// При нахождении любого хита сразу все попали в непрозрачный Face.
		// ray->tfar = -std::numeric_limits<float>::infinity();
		ctxt->energy = 0;
		args->valid[0] = -1;
		ctxt->Transparent = false;
	}
	else
	{
		ctxt->Transparent = true;
		args->valid[0] = -1;
	}
}
 
void FilterRaytraceTransparent(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;
	args->valid[0] = 0;

	if (hit->primID == RTC_INVALID_GEOMETRY_ID || hit->primID == ctxt->LastPremitive)
		return;

	ctxt->hits++;
	ctxt->LastPremitive = hit->primID;

	ray->org_x = ray->org_x + (ray->dir_x * ray->tfar);
	ray->org_y = ray->org_y + (ray->dir_y * ray->tfar);
	ray->org_z = ray->org_z + (ray->dir_z * ray->tfar);

	ray->tnear = TNearParram;
 
	// Перемещаем начало луча немного дальше пересечения
 	base_Face* F = (base_Face*)(TriTransparent_Dummys[hit->primID]); 	
	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];
  
	// barycentric coords
	// note: W,U,V order
	ctxt->B.set(1.0f - hit->u - hit->v, hit->u, hit->v);

	// calc UV
	Fvector2* cuv = F->getTC0();
	Fvector2	uv;
	uv.x = cuv[0].x * ctxt->B.x + cuv[1].x * ctxt->B.y + cuv[2].x * ctxt->B.z;
	uv.y = cuv[0].y * ctxt->B.x + cuv[1].y * ctxt->B.y + cuv[2].y * ctxt->B.z;

	int U = iFloor(uv.x * float(T.dwWidth) + .5f);
	int V = iFloor(uv.y * float(T.dwHeight) + .5f);
	U %= T.dwWidth;		if (U < 0) U += T.dwWidth;
	V %= T.dwHeight;	if (V < 0) V += T.dwHeight;
	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel = raw[V * T.dwWidth + U];
	u32 pixel_a = color_get_A(pixel);
	float opac = 1.f - _sqr(float(pixel_a) / 255.f);

	// Дополнение Контекста
	ctxt->energy *= opac;

	// Energy Loose
	if (ctxt->energy < 0.1f)
	{
		ray->tfar = -std::numeric_limits<float>::infinity();
		args->valid[0] = -1;
	}
}

#else 
void FilterRaytrace(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;

	base_Face* F = 0;

	args->valid[0] = 0;

	if (hit->geomID == 1)
	{
		F = (base_Face*)(TriTransparent_Dummys[hit->primID]);
	}
	else
	{
		F = (base_Face*)(TriNormal_Dummys[hit->primID]);
	}

	// Access to texture
	if (F->flags.bOpaque)
	{
		// При нахождении любого хита сразу все попали в непрозрачный Face.
		ray->tfar = -std::numeric_limits<float>::infinity();
		ctxt->energy = 0;
		args->valid[0] = 1;
		return;
	}

	// Перемещаем начало луча немного дальше пересечения
	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];

	if (T.pSurface.Empty())
		return;

	// barycentric coords
	// note: W,U,V order
	ctxt->B.set(1.0f - hit->u - hit->v, hit->u, hit->v);

	// calc UV
	Fvector2* cuv = F->getTC0();
	Fvector2	uv;
	uv.x = cuv[0].x * ctxt->B.x + cuv[1].x * ctxt->B.y + cuv[2].x * ctxt->B.z;
	uv.y = cuv[0].y * ctxt->B.x + cuv[1].y * ctxt->B.y + cuv[2].y * ctxt->B.z;

	int U = iFloor(uv.x * float(T.dwWidth) + .5f);
	int V = iFloor(uv.y * float(T.dwHeight) + .5f);
	U %= T.dwWidth;		if (U < 0) U += T.dwWidth;
	V %= T.dwHeight;	if (V < 0) V += T.dwHeight;
	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel = raw[V * T.dwWidth + U];
	u32 pixel_a = color_get_A(pixel);
	float opac = 1.f - _sqr(float(pixel_a) / 255.f);

	// Дополнение Контекста
	ctxt->energy *= opac;

	// Energy Loose
	//ray->tfar = -std::numeric_limits<float>::infinity();
	//args->valid[0] = 1;

}
#endif

FORCEINLINE void OcludedOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{
 	rtcInitIntersectContext(&data_hits.context);
 
	RTCRay rayhit;
	//SetRay1(&ray, rayhit);
	SetRay1Invert(&ray, rayhit);


	data_hits.StartPos = ray.pos;
	rtcOccluded1(IntelScene, &data_hits.context, &rayhit);

	// clMsg("Start: {%f, %f, %f}, End: {%f, %f, %f}", VPUSH(ray.pos), rayhit.org_x, rayhit.org_y, rayhit.org_z);
}

FORCEINLINE void RatraceOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{
 	rtcInitIntersectContext(&data_hits.context); 
	RTCRayHit rayhit;
	// SetRay1(&ray, rayhit);
	SetRay1Invert(&ray, rayhit);
	
	data_hits.StartPos = ray.pos;
	rtcIntersect1(IntelScene, &data_hits.context, &rayhit);
}

float RaytraceEmbreeProcess(CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& N, float range, Face* skip)
{
	/*
	float _u,_v, R;

	bool res = CDB::TestRayTri(P, N, L.tri, _u,_v, R, false);
	if (res)
	if (range > 0 && range < R)
		return 0;
	*/

	RayQueryContext data;
	data.Light = &L;
	// data.model = MDL;
	data.skip = skip;
	data.energy = 1.0f;
	data.Transparent = false;

	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = TNearParram;

	// Transparent
	data.Tfar = range;

	// Opacue Process
 	OcludedOneRay(ray, data);
//	if (data.Transparent)
//		RatraceOneRay(ray, data);
 	



	return data.energy;
}
  
constexpr double ShadowEpsilon = 1e-3f;
constexpr double AngleEpsilon = 1e-4f;

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	clMsg("error %d: %s", error, str);
	DebugBreak();
}

// OFF PACKED PROCESSING
void GetEmbreeDeviceProperty(LPCSTR msg, RTCDevice& device, RTCDeviceProperty prop)
{
	clMsg("EmbreeDevProp: %s : %llu", msg, rtcGetDeviceProperty(device, prop));
}

void IntelEmbreeSettings(bool avx, bool sse)
{
	string128 phase;
	sprintf(phase, "Intilized Intel Embree v3.15.5 - %s", avx ? "avx" : sse ? "sse" : "default");
	Phase(phase);

	TNearParram = build_args->embree_tnear;

	// CHECK THIS (Ускоряет ли)
	if (build_args->use_RobustGeom)
		rtcSetSceneFlags(IntelScene, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_ROBUST);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED", device, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED", device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED", device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED);

}

void InitializeGeometryAttach_CDB(RTCScene& scene)
{
	SpecialArgsXRLCLight::EmbreeGeom geom_type = (SpecialArgsXRLCLight::EmbreeGeom)build_args->embree_geometry_type;

	Fvector* CDB_verts = inlc_global_data()->RCAST_Model()->get_verts();
	CDB::TRI* CDB_tris = inlc_global_data()->RCAST_Model()->get_tris();

	IntelGeometryNormal = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometryNormal, RTCBuildQuality(geom_type));
	// rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterIntersectionOne);

	int v_cnt = inlc_global_data()->RCAST_Model()->get_verts_count();
	int t_cnt = inlc_global_data()->RCAST_Model()->get_tris_count();

	verticesNormal = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), v_cnt);
	trianglesNormal = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), t_cnt);

	SizeTriangleNormal = t_cnt;
	SizeVertexNormal = v_cnt;

	size_t VertexIndexer = 0;
	// FIX
	TriNormal_Dummys.clear();
	TriNormal_Dummys.reserve(inlc_global_data()->RCAST_Model()->get_tris_count());
	CDB::TRI* tri = inlc_global_data()->RCAST_Model()->get_tris();
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
		trianglesNormal[i].SetVertexes(CDB_verts, verticesNormal, VertexIndexer);
		TriNormal_Dummys[i] = (tri[i].pointer);
	}

	rtcCommitGeometry(IntelGeometryNormal);
	clMsg("[Intel Embree] Attached Geometry: IntelGeometry(Normal) By ID: %d", rtcAttachGeometry(scene, IntelGeometryNormal));
}


void InitializeGeometryAttach(bool Transparent, RTCScene& scene)
{
	SpecialArgsXRLCLight::EmbreeGeom geom_type = (SpecialArgsXRLCLight::EmbreeGeom)build_args->embree_geometry_type;

	Fvector* CDB_verts = inlc_global_data()->RCAST_Model()->get_verts();
	CDB::TRI* CDB_tris = inlc_global_data()->RCAST_Model()->get_tris();


	// Добавление вершин
	// 1я стадия подсчет того что можно для Embree Occluded

	// Устанавливать обезательно иле будет в Колбеке PrimID	= 0 
	if (!Transparent)
	{
		IntelGeometryNormal = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		rtcSetGeometryBuildQuality(IntelGeometryNormal, RTCBuildQuality(geom_type));
#ifdef USE_OCCLUSION
		rtcSetGeometryOccludedFilterFunction(IntelGeometryNormal, &FilterOcclusion);
 		//rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterRaytraceTransparent);
#else 
		rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterRaytrace);
#endif
	}
	else
	{
		IntelGeometryTransparent = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		rtcSetGeometryBuildQuality(IntelGeometryTransparent, RTCBuildQuality(geom_type));
#ifdef USE_OCCLUSION
		rtcSetGeometryOccludedFilterFunction(IntelGeometryTransparent, &FilterOcclusion);
		rtcSetGeometryIntersectFilterFunction(IntelGeometryTransparent, &FilterRaytraceTransparent);
#else 
		rtcSetGeometryIntersectFilterFunction(IntelGeometryTransparent, &FilterRaytrace);
#endif
	}

	// Буферы

	xr_vector<CDB::TRI*> TempBuffer;
	TempBuffer.clear();

	size_t Ignored_ByMissingTextures = 0;
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
		base_Face* F = (base_Face*)(CDB_tris[i].pointer);

		// Отсеиваем нахрен не нужное
		if (!F->Shader().flags.bLIGHT_CastShadow)
			continue;

		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];
		if (T.pSurface.Empty())
		{
			Ignored_ByMissingTextures++;
			F->flags.bOpaque = true;
		}

		if (Transparent && F->flags.bOpaque)
			continue;

		if (!Transparent && !F->flags.bOpaque)
			continue;

		TempBuffer.push_back(&CDB_tris[i]);
	}
 
	/*
	//string256 tmp;
	//sprintf(tmp, "[Intel Embree] Создаем Буфер под Треугольники c прозрачностью = (%d)", Transparent);
	//string256 tmp2;
	//sprintf(tmp2, "[Intel Embree] Геометрия: Треугольников: %lu, Вертексов: %lu, Проигнорировано изза DXT1: %d", TempBuffer.size(), TempBuffer.size() * 3, Ignored_ByMissingTextures);
	//clMsg(xr_string(tmp).c_str());
	//clMsg(xr_string(tmp2).c_str());
	*/

	// 2я Стадия Добавление 
	if (!Transparent)
	{
		verticesNormal = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), size_t(TempBuffer.size() * (3)));
		trianglesNormal = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), TempBuffer.size());

		SizeTriangleNormal = TempBuffer.size();
		SizeVertexNormal = TempBuffer.size() * 3;
	}
	else
	{
		verticesTransparent = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometryTransparent, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), size_t(TempBuffer.size() * (3)));
		trianglesTransparent = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometryTransparent, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), TempBuffer.size());

		SizeTriangleTransparent = TempBuffer.size();
		SizeVertexTransparent = TempBuffer.size() * 3;
	}

	// DUMMY BUFFER
	if (Transparent)
	{
		TriTransparent_Dummys.clear();
		TriTransparent_Dummys.reserve(TempBuffer.size());
	}
	else
	{
		TriNormal_Dummys.clear();
		TriNormal_Dummys.reserve(TempBuffer.size());
	}

	size_t VertexIndexer = 0;
	for (int i = 0; i < TempBuffer.size(); i++)
	{
		if (Transparent)
		{
			trianglesTransparent[i].SetVertexes(CDB_verts, verticesTransparent, VertexIndexer);
			TriTransparent_Dummys[i] = (TempBuffer[i]->pointer);
		}
		else
		{
			trianglesNormal[i].SetVertexes(CDB_verts, verticesNormal, VertexIndexer);
			TriNormal_Dummys[i] = (TempBuffer[i]->pointer);
		}
	}

	if (!Transparent)
	{
		rtcCommitGeometry(IntelGeometryNormal);
		clMsg("[Intel Embree] Attached Geometry: IntelGeometry(Normal) By ID: %d", rtcAttachGeometry(scene, IntelGeometryNormal));
	}
	else
	{
		rtcCommitGeometry(IntelGeometryTransparent);
		clMsg("[Intel Embree] Attached Geometry: IntelGeometry(Transparent) By ID: %d", rtcAttachGeometry(scene, IntelGeometryTransparent));
	}


	clMsg(xr_string("[Intel Embree] Создание Буфера закончено. ").c_str());
}

void IntelEmbereLOAD()
{
	bool avx = build_args->use_avx;
	bool sse = build_args->use_sse;
	char* config = avx ? "threads=16,isa=avx2" : sse ? "threads=16,isa=sse4.2" : "threads=16,isa=sse2";


	device = rtcNewDevice(config);
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
	IntelEmbreeSettings(avx, sse);

	// Создание сцены и добавление геометрии
	// Scene
	IntelScene = rtcNewScene(device);
	//IntelSceneTransparent = rtcNewScene(device);

	InitializeGeometryAttach(false, IntelScene); // Обычный буфер
	InitializeGeometryAttach(true, IntelScene);	 // Прозрачный буфер

	rtcCommitScene(IntelScene);
	//rtcCommitScene(IntelSceneTransparent);
}

void IntelEmbereUNLOAD()
{
	rtcReleaseGeometry(IntelGeometryNormal);
	rtcReleaseGeometry(IntelGeometryTransparent);

	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}


