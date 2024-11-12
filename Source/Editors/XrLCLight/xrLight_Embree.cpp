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
 
#pragma comment(lib, "embree4.lib")
#pragma comment(lib, "tbb.lib")
#include "embree4/rtcore.h"
  
RTCScene IntelScene;
RTCScene IntelSceneTransp;

RTCGeometry IntelGeometryNormal;
RTCGeometry IntelGeometryTransp;

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
		x = vertex.x;
		y = vertex.y;
		z = vertex.z;
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
};

/** NORMAL GEOM **/
VertexEmbree* verticesNormal = 0;
TriEmbree* trianglesNormal = 0;
u32 SizeTriangleNormal = 0;
size_t SizeVertexNormal = 0;
xr_vector<void*> TriNormal_Dummys;

/** TRANSPARENT **/
VertexEmbree* vertices_transp = 0;
TriEmbree* triangles_transp = 0;
u32 SizeTriangle_transp = 0;
size_t SizeVertex_transp = 0;
xr_vector<void*> TriTransp_Dummys;
 
 
RTCDevice device;

// ВАЖНЫЙ ПАРАМЕТР TNEAR Для пересечения с водой
int TNearParram = 0.2f;

void SetRay1(RayOptimizedCPU* ray, RTCRay& rayhit)
{
	rayhit.dir_x = ray->dir.x;
	rayhit.dir_y = ray->dir.y;
	rayhit.dir_z = ray->dir.z;

	rayhit.org_x = ray->pos.x;
	rayhit.org_y = ray->pos.y;
	rayhit.org_z = ray->pos.z;

	rayhit.tnear = ray->tmin;
	rayhit.tfar = ray->tmax;

	rayhit.mask = (unsigned int)(-1);
	rayhit.flags = 0;
}
 
void SetRay1(RayOptimizedCPU* ray, RTCRayHit& rayhit)
{
	rayhit.ray.dir_x = ray->dir.x;
	rayhit.ray.dir_y = ray->dir.y;
	rayhit.ray.dir_z = ray->dir.z;

	rayhit.ray.org_x = ray->pos.x;
	rayhit.ray.org_y = ray->pos.y;
	rayhit.ray.org_z = ray->pos.z;

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
	RTCRayQueryContext context;
	Fvector B;

 	Face* skip = 0;
	R_Light* Light = 0;
	float energy = 1.0f;
 
	u32 LastPremitive = 0;
	bool FindTransparent = false;
	int HitsCnt = 0;
};

void FilterRaytrace(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;
	
 	if (hit->primID == RTC_INVALID_GEOMETRY_ID)
		return;

	base_Face* F = (base_Face*)(TriNormal_Dummys[hit->primID]);
	
	args->valid[0] = 0;

	//if (hit->geomID == 0)
  	if (!F->flags.bOpaque)
	{
		// Перемещаем начало луча немного дальше пересечения
		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];

 		// barycentric coords
		// note: W,U,V order
		ctxt->B.set(1.0f - hit->u - hit->v, hit->u, hit->v);
 
		
		//// calc UV
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

		if (ctxt->energy < 0.1f)
		{
			ray->tfar = -std::numeric_limits<float>::infinity();
			ctxt->energy = 0;
			args->valid[0] = -1;	// Стоп для поиска дальнейшого 
			return;
		}
	}

 	if (F->flags.bOpaque)
	{
		// При нахождении любого хита сразу все попали в непрозрачный Face.
		ray->tfar = -std::numeric_limits<float>::infinity();
		ctxt->energy = 0;
		args->valid[0] = -1;	// Стоп для поиска дальнейшого 
		return;
	}

}

float RaytraceEmbreeProcess(CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& N, float range, Face* skip)
{
  	RayQueryContext data_hits;
	data_hits.Light = &L;
	data_hits.skip  = skip;
	data_hits.energy = 1.0f;
      			
	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = TNearParram;
  
	// Start Raytrce buffer
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);
	data_hits.context = context;

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.context = &data_hits.context;

	RTCRayHit rayhit;
	SetRay1(&ray, rayhit);

	rtcIntersect1(IntelScene, &rayhit, &args);

	return data_hits.energy;
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
	sprintf(phase, "Intilized Intel Embree v4.1.0 - %s", avx ? "avx" : sse ? "sse" : "default");
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
	
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_TASKING_SYSTEM", device, RTC_DEVICE_PROPERTY_TASKING_SYSTEM);
}


void InitializeGeometryAttach_CDB(RTCScene& scene)
{
	SpecialArgsXRLCLight::EmbreeGeom geom_type = (SpecialArgsXRLCLight::EmbreeGeom)build_args->embree_geometry_type;

	Fvector* CDB_verts = inlc_global_data()->RCAST_Model()->get_verts();
	CDB::TRI* CDB_tris = inlc_global_data()->RCAST_Model()->get_tris();

	IntelGeometryNormal = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometryNormal, RTCBuildQuality(geom_type));
	
	//rtcSetGeometryOccludedFilterFunction(IntelGeometryNormal, &FilterRaytraceOcc);
	rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterRaytrace);


	int v_cnt = inlc_global_data()->RCAST_Model()->get_verts_count();
	int t_cnt = inlc_global_data()->RCAST_Model()->get_tris_count();

	verticesNormal = (VertexEmbree*)	rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), v_cnt);
	trianglesNormal = (TriEmbree*)		rtcSetNewGeometryBuffer(IntelGeometryNormal, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), t_cnt);
 
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
 	InitializeGeometryAttach_CDB(IntelScene);
 	rtcCommitScene(IntelScene); 
}

void IntelEmbereUNLOAD()
{
	rtcReleaseGeometry(IntelGeometryNormal);
 	rtcReleaseScene(IntelScene);
   
 	rtcReleaseDevice(device);
}
