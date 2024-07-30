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
// RTCScene IntelSceneTransparent;

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


VertexEmbree* verticesNormal = 0;
TriEmbree* trianglesNormal = 0;
u32 SizeTriangleNormal = 0;
size_t SizeVertexNormal = 0;

VertexEmbree* verticesTransparent = 0;
TriEmbree* trianglesTransparent   = 0;
u32 SizeTriangleTransparent = 0;
size_t SizeVertexTransparent = 0;

xr_vector<void*> TriNormal_Dummys;
xr_vector<void*> TriTransparent_Dummys;




RTCDevice device;

// ВАЖНЫЙ ПАРАМЕТР TNEAR Для пересечения с водой
int TNearParram = 0.2f;

  
/*
void SetRay8(RayOptimizedCPU* ray, RTCRayHit8& rayhit, int ray_id)
{
	rayhit.ray.org_x[ray_id] = ray->pos.x;
	rayhit.ray.org_y[ray_id] = ray->pos.y;
	rayhit.ray.org_z[ray_id] = ray->pos.z;

	rayhit.ray.dir_x[ray_id] = ray->dir.x;
	rayhit.ray.dir_y[ray_id] = ray->dir.y;
	rayhit.ray.dir_z[ray_id] = ray->dir.z;

	rayhit.ray.flags[ray_id] = 0;
	rayhit.ray.mask[ray_id] = -1;

	rayhit.ray.tfar[ray_id] = 1000.0f;
	rayhit.ray.tnear[ray_id] = 0.0f;

	rayhit.hit.u[ray_id] = 0;
	rayhit.hit.v[ray_id] = 0;

	rayhit.hit.Ng_x[ray_id] = 0;
	rayhit.hit.Ng_y[ray_id] = 0;
	rayhit.hit.Ng_z[ray_id] = 0;

	rayhit.hit.primID[ray_id] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.geomID[ray_id] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0][ray_id] = RTC_INVALID_GEOMETRY_ID;
}

void ReSetRay8(RTCRayHit8& rayhit, int ray_id)
{
	rayhit.ray.tnear[ray_id] = rayhit.ray.tfar[ray_id] + 0.01f;
	rayhit.ray.tfar[ray_id] = 1000.0f;

	rayhit.hit.u[ray_id] = 0;
	rayhit.hit.v[ray_id] = 0;

	rayhit.hit.Ng_x[ray_id] = 0;
	rayhit.hit.Ng_y[ray_id] = 0;
	rayhit.hit.Ng_z[ray_id] = 0;

	rayhit.hit.primID[ray_id] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.geomID[ray_id] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0][ray_id] = RTC_INVALID_GEOMETRY_ID;
}

RTCRayHit GetRay8(RTCRayHit8& ray_i, size_t i)
{
	RTCRayHit ray_o;
	ray_o.ray.org_x = ray_i.ray.org_x[i];
	ray_o.ray.org_y = ray_i.ray.org_y[i];
	ray_o.ray.org_z = ray_i.ray.org_z[i];
	ray_o.ray.dir_x = ray_i.ray.dir_x[i];
	ray_o.ray.dir_y = ray_i.ray.dir_y[i];
	ray_o.ray.dir_z = ray_i.ray.dir_z[i];
	ray_o.ray.tnear = ray_i.ray.tnear[i];
	ray_o.ray.tfar = ray_i.ray.tfar[i];
	ray_o.ray.time = ray_i.ray.time[i];
	ray_o.ray.mask = ray_i.ray.mask[i];
	ray_o.ray.id = ray_i.ray.id[i];
	ray_o.hit.geomID = ray_i.hit.geomID[i];
	ray_o.hit.primID = ray_i.hit.primID[i];
	ray_o.hit.u = ray_i.hit.u[i];
	ray_o.hit.v = ray_i.hit.v[i];
	ray_o.hit.Ng_x = ray_i.hit.Ng_x[i];
	ray_o.hit.Ng_y = ray_i.hit.Ng_y[i];
	ray_o.hit.Ng_z = ray_i.hit.Ng_z[i];

	for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
		ray_o.hit.instID[l] = ray_i.hit.instID[l][i];

	return ray_o;
}
*/

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

void SetRay1Hit(RTCRayHit& rayhit, float range = 0)
{
	rayhit.ray.tnear = rayhit.ray.tfar + 0.01f;
	rayhit.ray.tfar = range == 0 ? 1000.0f : range;

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

// 1 - 8 - 32 - 64 - 96 - 128 - 256 - 512 - 1024
//#define MAX_HITS 1024 
 
 
struct DataFaceGlobalE
{
	base_Face* face = 0;
	RTCRayHit* rayhit = 0;
	float energy;
	float tfar;
};

struct RayQueryContext
{
	RTCRayQueryContext context;
	Fvector B;

	// CDB::MODEL* model = 0; 
	Face* skip = 0;
	R_Light* Light = 0;
	 
	float energy = 1.0f;
 	int hits = 0;
	bool Opacity = false;
};

 
struct RayQueryContext8
{
	RTCRayQueryContext context;

	CDB::MODEL* model;
	Face* skip[8];
	R_Light* Light = 0;
	Fvector B;

	int count = 0;
	float energy = 1.0f;
 };

void SetRayHit8(RTCRayHit8& rayhit8, PackedBuffer* buffer)
{
	for (int i = 0; i < 8; i++)
	{
		rayhit8.ray.dir_x[i] = buffer->dir[i].x;
		rayhit8.ray.dir_y[i] = buffer->dir[i].y;
		rayhit8.ray.dir_z[i] = buffer->dir[i].z;

		rayhit8.ray.org_x[i] = buffer->pos[i].x;
		rayhit8.ray.org_y[i] = buffer->pos[i].y;
		rayhit8.ray.org_z[i] = buffer->pos[i].z;

		rayhit8.ray.mask[i] = (unsigned int)(-1);
		rayhit8.ray.flags[i] = 0;

		rayhit8.hit.Ng_x[i] = 0;
		rayhit8.hit.Ng_y[i] = 0;
		rayhit8.hit.Ng_z[i] = 0;

		rayhit8.hit.u[i] = 0;
		rayhit8.hit.v[i] = 0;

		rayhit8.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		rayhit8.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		rayhit8.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;


		rayhit8.ray.tfar[i] = buffer->tmax[i];
		rayhit8.ray.tnear[i] = 0.f;

	}
}


xrCriticalSection csLIGHT;
 
#define USE_OCCLUSION


#ifdef USE_OCCLUSION
void FilterOcclusion(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
 	RTCRay* ray = (RTCRay*)args->ray;

	// При нахождении любого хита сразу все попали в непрозрачный Face.
	ray->tfar = -std::numeric_limits<float>::infinity();
	ctxt->energy = 0;
	args->valid[0] = 0;
}
#else 
void FilterRaytrace(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;
 
	if (hit->geomID != 0)
		return;

	// Продолжать собирать 
	ray->tfar = -std::numeric_limits<float>::infinity();
	ctxt->energy = 0;
	args->valid[0] = 1;
}
#endif


void FilterRaytraceTransparent(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;

//	if (hit->geomID != 1)
//		return;
	 
	// Access to texture
	base_Face* F = (base_Face*)(TriTransparent_Dummys[hit->primID]);

	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];
 
	if (T.pSurface.Empty())
	{
		Msg("HITS Starge detected: geom: %d, primID: %lu, intsPrim: %lu, sizeArrRay: %lu", hit->geomID, hit->primID, hit->instPrimID, TriTransparent_Dummys.size());
		return;
	}

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
	ctxt->hits++;

	// Energy Loose
	if (ctxt->energy <= 0.001f)
	{
		ray->tfar = -std::numeric_limits<float>::infinity();
		args->valid[0] = 1;
		ctxt->energy = 0;
	}
 
}
	 
FORCEINLINE void OcludedOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);
	data_hits.context = context;

	RTCOccludedArguments args;
	rtcInitOccludedArguments(&args);
	// args.filter = &FilterOcludedOne;
	args.context = &data_hits.context;
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER ); /*| RTC_RAY_QUERY_FLAG_COHERENT*/

	RTCRay rayhit;
	SetRay1(&ray, rayhit);

 	rtcOccluded1(IntelScene, &rayhit, &args);
}
 
FORCEINLINE void RatraceOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{ 
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);
	data_hits.context = context;
 
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
 	args.context = &data_hits.context;	 
 
	RTCRayHit rayhit;
	SetRay1(&ray, rayhit);

    rtcIntersect1(IntelScene, &rayhit, &args);    
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
	data.skip  = skip;
	data.energy = 1.0f;
     			
	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = TNearParram;
	
	OcludedOneRay(ray, data);
 	if (data.energy != 0)	// Процесс для Transparents
		RatraceOneRay(ray, data);
		
	return data.energy;
}
 

void FilterIntersection8(const struct RTCFilterFunctionNArguments* args)
{
	/*

	//for (unsigned int i = 0; i < args->N; i++)
	{
		if (args->valid[i] != -1)
			continue;


		//if (RTCHitN_primID(args->hit, args->N, i) & 2)
		//{
		//	args->valid[i] = 0;
		//}

		rtcGetRayFromRayN(args->ray, args->N, i);
		rtcGetRayFromRayN(args->hit, i);


	}
	*/

	clMsg("Ray: %d", args->N);
}

void Raytrace8Ray(PackedBuffer* buffer, RayQueryContext8& data_hits)
{
	clMsg("Raytrace 8 Rays");

	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	data_hits.context = context;

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.filter = &FilterIntersection8;
	args.context = &data_hits.context;
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER /*| RTC_RAY_QUERY_FLAG_COHERENT*/);

	RTCRayHit8 rayhit8;
	SetRayHit8(rayhit8, buffer);

	rtcIntersect8(buffer->valid, IntelScene, &rayhit8, &args);
}

void RayTraceEmbree8Preocess(PackedBuffer* buffer, ELightType type_lightpoint, ELights type_LIGHTs)
{
 	/* RGB */
	/*
	if (LT_DIRECT == type_lightpoint)
	{
		// Trace Light
		//float scale = D * L->energy * rayTrace(DB, MDL, *L, Pnew, Ldir, 1000.f, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode);


		float scale[8];

		for (auto i = 0; i < 8; i++)
		{
			buffer->tmax[i] = 1000.0f;
		}

		SetRayHit8(rayhit8, buffer, type_lightpoint);

		for (auto i = 0; i < 8; i++)
		{
			buffer->color[i].rgb.x += scale[i] * buffer->light->diffuse.x;
			buffer->color[i].rgb.y += scale[i] * buffer->light->diffuse.y;
			buffer->color[i].rgb.z += scale[i] * buffer->light->diffuse.z;
		}


		//C.rgb.x += scale * L->diffuse.x;
		//C.rgb.y += scale * L->diffuse.y;
		//C.rgb.z += scale * L->diffuse.z;

	}

	if (LT_POINT == type_lightpoint)
	{
		// Trace Light
		float scale[8]; //= D * L->energy * rayTrace(DB, MDL, *L, Pnew, Ldir, R, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode);
		float A[8];
		float R[8];

		for (auto i = 0; i < 8; i++)
		{
			R[i]  = _sqrt(buffer->Dist2Light[i]);
			buffer->tmax[i] = R[i];

			/// buffer->MDL = MDL;
			//DB, MDL, * L, Pnew, Ldir, R, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode)


			//	SetRayHit8(rayhit8, buffer, type_lightpoint);

			if (inlc_global_data()->gl_linear())
			{
				A[i] = 1 - R[i] / buffer->light->range;
			}
			else
			{
				//	Igor: let A equal 0 at the light boundary
				A[i] = scale[i] *
				( 1 / (
					buffer->light[i].attenuation0 +
					buffer->light[i].attenuation1 * R[i] +
					buffer->light[i].attenuation2 * buffer->Dist2Light[i]) -

					R[i] * buffer->light[i].falloff // LAST MININUS
				);

			}

			buffer->color[i].rgb.x += A[i] * buffer->light->diffuse.x;
			buffer->color[i].rgb.y += A[i] * buffer->light->diffuse.y;
			buffer->color[i].rgb.z += A[i] * buffer->light->diffuse.z;
		}

		//SetRayHit8(rayhit8, buffer, type_lightpoint);
	}

	if (LT_SECONDARY == type_lightpoint)
	{
		/*
			// Jitter + trace light -> monte-carlo method
			Fvector	Psave = L->position, Pdir;
			L->position.mad(Pdir.random_dir(L->direction, PI_DIV_4), .05f);

			float R = _sqrt(sqD);
			float scale = powf(D, 1.f / 8.f) * L->energy * rayTrace(DB, MDL, *L, Pnew, Ldir, R, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode);
			float A = scale * (1 - R / L->range);
			L->position = Psave;

			C.rgb.x += A * L->diffuse.x;
			C.rgb.y += A * L->diffuse.y;
			C.rgb.z += A * L->diffuse.z;
		*
	}
	*/

	/* HEMI */

	if (type_LIGHTs == ELights::Hemi)
	{
		switch (type_lightpoint)
		{
		case LT_Direct:
		{
			//float scale = L->energy * rayTrace(DB, MDL, *L, PMoved, Ldir, 1000.f, skip);
			//C.hemi += scale;
			RayQueryContext8 data;
			data.Light = buffer->light;
			data.model = buffer->MDL;
			for (auto i = 0; i < 8; i++)
				data.skip[i] = buffer->skip[i];
			data.energy = 1.0f;
			data.count = 0;

			Raytrace8Ray(buffer, data);


		}
		break;

		case LT_Point:
		{
			//float scale = D * L->energy * rayTrace(DB, MDL, *L, Pnew, Ldir, R, skip, bUseFaceDisable);
			//float A = scale / (L->attenuation0 + L->attenuation1 * R + L->attenuation2 * sqD);

			//C.hemi += A;

			RayQueryContext8 data;
			data.Light = buffer->light;
			data.model = buffer->MDL;
			for (auto i = 0; i < 8; i++)
				data.skip[i] = buffer->skip[i];
			data.energy = 1.0f;
			data.count = 0;

			Raytrace8Ray(buffer, data);


		}
		break;

		default:
			break;
		}
	}

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
#else 
		rtcSetGeometryIntersectFilterFunction(IntelGeometryNormal, &FilterRaytrace);
#endif
 	}
	else
	{
		IntelGeometryTransparent = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		rtcSetGeometryBuildQuality(IntelGeometryTransparent, RTCBuildQuality(geom_type));
		rtcSetGeometryIntersectFilterFunction(IntelGeometryTransparent, &FilterRaytraceTransparent);
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


	string256 tmp;
	sprintf(tmp, "[Intel Embree] Создаем Буфер под Треугольники c прозрачностью = (%d)", Transparent);
	string256 tmp2;
	sprintf(tmp2, "[Intel Embree] Геометрия: Треугольников: %lu, Вертексов: %lu, Проигнорировано изза DXT1: %d", TempBuffer.size(), TempBuffer.size() * 3, Ignored_ByMissingTextures); 
	clMsg(xr_string(tmp).c_str());
	clMsg(xr_string(tmp2).c_str());
	 

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
//	IntelSceneTransparent = rtcNewScene(device);

	InitializeGeometryAttach(false, IntelScene); // Обычный буфер
	InitializeGeometryAttach(true, IntelScene);	 // Прозрачный буфер

	rtcCommitScene(IntelScene);
//	rtcCommitScene(IntelSceneTransparent);

	/*
	RTCBounds bounds;
	rtcGetSceneBounds(IntelScene, &bounds);
 	clMsg("SceneBounds: [%f][%f][%f] max [%f][%f][%f] a0: %f, a1: %f",
		bounds.lower_x, bounds.lower_y, bounds.lower_z,
		bounds.upper_x, bounds.upper_y, bounds.upper_z,
		bounds.align0, bounds.align1);
  
	 

	
	Fvector* CDB_verts = inlc_global_data()->RCAST_Model()->get_verts();
	CDB::TRI* CDB_tris = inlc_global_data()->RCAST_Model()->get_tris();

	Fbox bb_base;
 	bb_base.null();

	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
		auto v1 = CDB_verts[inlc_global_data()->RCAST_Model()->get_tris()[i].verts[0]];
		auto v2 = CDB_verts[inlc_global_data()->RCAST_Model()->get_tris()[i].verts[0]];
		auto v3 = CDB_verts[inlc_global_data()->RCAST_Model()->get_tris()[i].verts[0]];

		bb_base.modify(v1);
		bb_base.modify(v2);
		bb_base.modify(v3);
	}

	clMsg("CDB SceneBounds: [%f][%f][%f] max [%f][%f][%f]",
		bb_base.min.x, bb_base.min.y, bb_base.min.z, 
		bb_base.max.x, bb_base.max.y, bb_base.max.z);
	*/

}

void IntelEmbereUNLOAD()
{
	rtcReleaseGeometry(IntelGeometryNormal);
	rtcReleaseGeometry(IntelGeometryTransparent);

 	rtcReleaseScene(IntelScene);
 	rtcReleaseDevice(device);
}
 

 