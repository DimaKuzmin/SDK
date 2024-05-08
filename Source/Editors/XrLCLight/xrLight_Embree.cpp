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


xr_map<u32, bool> opacity;
xr_map<u32, bool> cast_shadowed;
       
xr_map<int, std::atomic<u64>> ticks_process;
xr_map<int, std::atomic<u64>> ticks_process_hits;
 
#pragma comment(lib, "embree4.lib")
#pragma comment(lib, "tbb.lib")
#include "embree4/rtcore.h"
 

RTCScene IntelScene;
RTCGeometry IntelGeometry;
RTCDevice device;

  
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

void SetRay1(RayOptimizedCPU* ray, RTCRayHit& rayhit)
{
	rayhit.ray.dir_x = ray->dir.x;
	rayhit.ray.dir_y = ray->dir.y;
	rayhit.ray.dir_z = ray->dir.z;

	rayhit.ray.org_x = ray->pos.x;
	rayhit.ray.org_y = ray->pos.y;
	rayhit.ray.org_z = ray->pos.z;

	rayhit.ray.tnear = 0.0f; 
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
 
 
struct DataFaceGlobal
{
	base_Face* face;
	float energy;
	float tfar;
};

struct RayQueryContext
{
	RTCRayQueryContext context;

	CDB::MODEL* model; 
	Face* skip;
	R_Light* Light;
	Fvector B;
	 	

	float energy = 1.0f;
	int Ended = 0;
 
	//float last_far = 1000.0f;
	unsigned int LastPrimitive = 0;

};

 
struct RayQueryContext8
{
	RTCRayQueryContext context;

	CDB::MODEL* model;
	Face* skip[8];
	R_Light* Light;
	Fvector B;

	int count = 0;
	float energy = 1.0f;
	float last_far = 1000.0f;
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

int RAY_ID = 0;
  
void FilterIntersectionOne(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*) args->context;

	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;
	
	if (ctxt->Ended || hit->primID == RTC_INVALID_GEOMETRY_ID || hit->geomID == RTC_INVALID_GEOMETRY_ID)
 		return;
 
	if (hit->primID == ctxt->LastPrimitive)
	{
		args->valid[0] = 0;
		return;
	}

	ctxt->LastPrimitive = hit->primID;

	// Access to texture
	CDB::TRI* clT = &ctxt->model->get_tris()[hit->primID];	 

	base_Face* F = (base_Face*)(clT->pointer);
 
	if (0 == F || ctxt->skip == F)
	{
		args->valid[0] = 0;
		return;
	}

	const Shader_xrLC& SH = F->Shader();
	if (!SH.flags.bLIGHT_CastShadow ) // || F->flags.bShadowSkip
	{
		args->valid[0] = 0;
		return;
	}

	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];
  
	if (F->flags.bOpaque)	 
	{ 
		// Opaque poly - cache it
		R_Light& light = (* ( (R_Light*) ctxt->Light) );
 
		// Opaque poly - cache it
		light.tri[0].set	(ctxt->model->get_verts()[clT->verts[0]]);
		light.tri[1].set	(ctxt->model->get_verts()[clT->verts[1]]);
		light.tri[2].set	(ctxt->model->get_verts()[clT->verts[2]]);
		
		args->valid[0] = -1;
		ctxt->energy = 0;
 		ctxt->Ended = 1;
 
   		return;
	}
 
	if ( T.pSurface.Empty() )
	{	 
		F->flags.bOpaque = true;
		args->valid[0] = -1;
		ctxt->energy = 0;
 		ctxt->Ended = 1;
 
		clMsg("* ERROR: RAY-TRACE: Strange face detected... Has alpha without texture...: %s", T.name);

  		return;
	}
 
	// barycentric coords
	// note: W,U,V order
 
	ctxt->B.set	(1.0f - hit->u - hit->v, hit->u, hit->v);

	// calc UV
	Fvector2*	cuv = F->getTC0					();
	Fvector2	uv;
	uv.x = cuv[0].x*ctxt->B.x + cuv[1].x*ctxt->B.y + cuv[2].x*ctxt->B.z;
	uv.y = cuv[0].y*ctxt->B.x + cuv[1].y*ctxt->B.y + cuv[2].y*ctxt->B.z;

	int U = iFloor(uv.x*float(T.dwWidth) + .5f);
	int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
	U %= T.dwWidth;	
	if (U<0) U+=T.dwWidth;
	V %= T.dwHeight;
	if (V<0) V+=T.dwHeight;

	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel		= raw[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
	ctxt->energy *= opac;  
  
	// Energy Loose
	if (ctxt->energy <= 0.001f)
	{
		args->valid[0] = -1;
		ctxt->energy = 0;
		ctxt->Ended = 1;
	}
	else
	{
		args->valid[0] = 0;
	}
}


extern u64 RayID;
xrCriticalSection csEmbree;

void RatraceOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{ 
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	data_hits.context = context;
 
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.filter = &FilterIntersectionOne;
	args.context = &data_hits.context;	 
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER /*| RTC_RAY_QUERY_FLAG_COHERENT*/);
	
	RTCRayHit rayhit;
	SetRay1(&ray, rayhit);
	
	RAY_ID++;
	rtcIntersect1(IntelScene, &rayhit, &args);    


}

float RaytraceEmbreeProcess(CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& N, float range, Face* skip)
{
  	float _u,_v, R;
	bool res = CDB::TestRayTri(P, N, L.tri, _u,_v, R, false);
	if (res) 
	if (range > 0 && range < R) 
 		return 0;
 
  	RayQueryContext data;
	data.Light = &L;
	data.model = MDL;
	data.skip  = skip;
	data.energy = 1.0f;
	data.Ended = 0;
   			
	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = 0;
	
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

	Msg("Ray: %d", args->N);
}

void Raytrace8Ray(PackedBuffer* buffer, RayQueryContext8& data_hits)
{
	Msg("Raytrace 8 Rays");

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
	Msg("error %d: %s", error, str);
}

// OFF PACKED PROCESSING
void GetEmbreeDeviceProperty(LPCSTR msg, RTCDevice& device, RTCDeviceProperty prop)
{
	Msg("EmbreeDevProp: %s : %llu", msg, rtcGetDeviceProperty(device, prop));
}

void IntelEmbereLOAD()
{
	std::string config;
	bool avx = false, sse = false;
	if (avx = strstr(Core.Params, "-use_avx"))
		config = "threads=8,isa=avx2";
	else if (sse = strstr(Core.Params, "-use_sse"))
		config = "threads=8,isa=sse4.2";
	else
		config = "threads=8,isa=sse2";

	device = rtcNewDevice(config.c_str());
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);

	
	Status("Intilized Intel Embree v4.1.0 - %s", avx ? "avx" : sse ? "sse" : "default");
	
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED", device, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED", device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED", device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED);


	// Создание сцены и добавление геометрии
	 
	struct VertexEmbree { float x, y, z; };
	VertexEmbree* vertices;
	struct TriEmbree { uint32_t point1, point2, point3; };
	TriEmbree* triangles;
	// Добавление вершин

	// Scene
	IntelScene = rtcNewScene(device); 
	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

 	// CHECK THIS (Ускоряет ли)
	
	if ( strstr(Core.Params, "-intel_quality_low") )
	{
		rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_LOW);
	}
	else
	{
		// CHECK THIS (Бьет ли по производительности)
		rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_REFIT );  
		rtcSetSceneFlags(IntelScene, RTC_SCENE_FLAG_COMPACT | RTC_SCENE_FLAG_ROBUST);
	}
	
	

 
	vertices = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), inlc_global_data()->RCAST_Model()->get_verts_count());
	triangles = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), inlc_global_data()->RCAST_Model()->get_tris_count());
 	 
	Msg("Intel Embree, Geometry 0: Vertices: %d", inlc_global_data()->RCAST_Model()->get_verts_count());
	Msg("Intel Embree, Geometry 0: Triangles: %d", inlc_global_data()->RCAST_Model()->get_tris_count());


	Fvector* vertex_CDB = inlc_global_data()->RCAST_Model()->get_verts();
	CDB::TRI* tri = inlc_global_data()->RCAST_Model()->get_tris();
 
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
 		triangles[i].point1 = tri[i].verts[0];  
		triangles[i].point2 = tri[i].verts[1];
		triangles[i].point3 = tri[i].verts[2];

		VertexEmbree& vert_1 = vertices[tri[i].verts[0]];
		VertexEmbree& vert_2 = vertices[tri[i].verts[1]];
		VertexEmbree& vert_3 = vertices[tri[i].verts[2]];

		Fvector& v1 = vertex_CDB[tri[i].verts[0]];
		Fvector& v2 = vertex_CDB[tri[i].verts[1]];
		Fvector& v3 = vertex_CDB[tri[i].verts[2]];

		vert_1.x = v1.x;
		vert_1.y = v1.y;
		vert_1.z = v1.z;

		vert_2.x = v2.x;
		vert_2.y = v2.y;
		vert_2.z = v2.z;

		vert_3.x = v3.x;
		vert_3.y = v3.y;
		vert_3.z = v3.z;
	}
				 
	rtcCommitGeometry(IntelGeometry);
	rtcAttachGeometry(IntelScene, IntelGeometry);
	
	// Check need
	rtcCommitScene(IntelScene);

	RTCBounds bounds;
	rtcGetSceneBounds(IntelScene, &bounds );
 
	Msg("SceneBounds: [%f][%f][%f] max [%f][%f][%f] a0: %f, a1: %f", bounds.lower_x, bounds.lower_y, bounds.lower_z, bounds.upper_x, bounds.upper_y, bounds.upper_z, bounds.align0, bounds.align1);
 
	// Устанавливать обезательно иле будет в Колбеке PrimID	= 0 
	rtcSetGeometryIntersectFilterFunction(IntelGeometry, &FilterIntersectionOne);	

 
}

void IntelEmbereUNLOAD()
{
	rtcReleaseGeometry(IntelGeometry);
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}

// END INTEL CODE

 