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

#include "embree4/rtcore.h"
#include "EmbreeDataStorage.h"
#include <atomic>

RTCScene IntelScene;
RTCGeometry IntelGeometry;
RTCDevice device;

xr_map<u32, bool> opacity;
xr_map<u32, bool> cast_shadowed;
       
xr_map<int, std::atomic<u64>> ticks_process;
xr_map<int, std::atomic<u64>> ticks_process_hits;


  
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

void SetRay1(RayOptimizedCPU* ray, RTCRayHit& rayhit)
{
	rayhit.ray.dir_x = ray->dir.x;
	rayhit.ray.dir_y = ray->dir.y;
	rayhit.ray.dir_z = ray->dir.z;

	rayhit.ray.org_x = ray->pos.x;
	rayhit.ray.org_y = ray->pos.y;
	rayhit.ray.org_z = ray->pos.z;

	rayhit.ray.tnear = 0.0f;
	rayhit.ray.tfar = 1000.0f;

	rayhit.ray.mask = -1;
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
   
 

void IntelClearTimers(LPCSTR name)
{
	ticks_process.clear();
	ticks_process_hits.clear();
}

// Сделать потом переключалку

// 1 - 8 - 32 - 64 - 96 - 128 - 256 - 512 - 1024
//#define MAX_HITS 1024 
 
#define EMBREE_V4

#ifdef EMBREE_V4
 
struct RayQueryContext
{
	RTCRayQueryContext context;

	CDB::MODEL* model; 
	Face* skip;
	R_Light* Light;
	Fvector B;
	 	
	int count = 0;
	float energy = 1.0f;
	xr_vector<Hits_XR> hits;

};

void FilterIntersectionOne(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*) args->context;
 
	RTCHit* hit = (RTCHit*) args->hit;
	RTCRay* ray = (RTCRay*) args->ray;
 
	args->valid[0] = 0;
	ctxt->count++;

	// Access to texture
	CDB::TRI* clT = &ctxt->model->get_tris()[hit->primID];	 
	base_Face* F = (base_Face*)(clT->pointer);
		
	if (nullptr == F)					return;	

	if (ctxt->skip == F)				return;

	const Shader_xrLC& SH = F->Shader();
	if (!SH.flags.bLIGHT_CastShadow)	return;

	if (F->flags.bOpaque)	 
	{ 
		// Opaque poly - cache it
		R_Light& light = (* ( (R_Light*) ctxt->Light) );
 
		// Opaque poly - cache it
		light.tri[0].set	(ctxt->model->get_verts()[clT->verts[0]]);
		light.tri[1].set	(ctxt->model->get_verts()[clT->verts[1]]);
		light.tri[2].set	(ctxt->model->get_verts()[clT->verts[2]]);
		
		
		// SE7KILLS CHECK THIS THING 
		//ctxt->Light->tri[0].set(ctxt->model->get_verts()[clT->verts[0]]);
		//ctxt->Light->tri[1].set(ctxt->model->get_verts()[clT->verts[1]]);
		//ctxt->Light->tri[2].set(ctxt->model->get_verts()[clT->verts[2]]);
 
		if (ray->tfar > 1.f)
		{
			args->valid[0] = -1;
			ctxt->energy = 0;
		}

  		return;
	}
 		
	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];

	if (T.pSurface.Empty() )
	{	 
		F->flags.bOpaque = true;
  		
		if (ray->tfar > 1.f)
		{
			args->valid[0] = -1;
			ctxt->energy = 0;
		}
		
		Msg("Texture: %s, tfar: %f null surface", T.name, ray->tfar);
		 
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
	if (U<0)
		U+=T.dwWidth;
	V %= T.dwHeight;	
	if (V<0) 
		V+=T.dwHeight;
	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel		= raw[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
	ctxt->energy *= opac;
}

void RatraceOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{ 
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	data_hits.context = context;
 
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.filter = &FilterIntersectionOne;
	args.context = &data_hits.context;	 
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER | RTC_RAY_QUERY_FLAG_COHERENT);
	
	RTCRayHit rayhit;
	SetRay1(&ray, rayhit);
	rtcIntersect1(IntelScene, &rayhit, &args);    
}
#else 

struct RayQueryContext
{
	RTCIntersectContext context;

	CDB::MODEL* model; 
	Face* skip;
	R_Light* Light;
	Fvector B;
	 	
	int count = 0;
	float energy = 1.0f;
	xr_vector<Hits_XR> hits;
};
   

 
void FilterIntersectionOne(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*) args->context;
 
	RTCHit* hit = (RTCHit*) args->hit;
	RTCRay* ray = (RTCRay*) args->ray;
 
	args->valid[0] = 0;


	// Access to texture
	CDB::TRI* clT = &ctxt->model->get_tris()[hit->primID];	 
	base_Face* F = (base_Face*)(clT->pointer);
		
	if (nullptr == F)					return;	
	if (ctxt->skip == F)				return;
	
	const Shader_xrLC& SH = F->Shader();
	if (!SH.flags.bLIGHT_CastShadow)	return;

	if (F->flags.bOpaque)	 
	{ 
		// Opaque poly - cache it
		// SE7KILLS CHECK THIS THING 
		ctxt->Light->tri[0].set(ctxt->model->get_verts()[clT->verts[0]]);
		ctxt->Light->tri[1].set(ctxt->model->get_verts()[clT->verts[1]]);
		ctxt->Light->tri[2].set(ctxt->model->get_verts()[clT->verts[2]]);
  		
		ctxt->energy = 0;  		
		args->valid[0] = -1; 	
		return;
	}
 
	b_material& M = inlc_global_data()->materials()[F->dwMaterial];
	b_texture& T = inlc_global_data()->textures()[M.surfidx];

	if (T.pSurface.Empty() )
	{	 
		F->flags.bOpaque = true;	
		
		ctxt->energy = 0;
		args->valid[0] = -1; 	
		Msg("Texture: %s, tfar: %f null surface", T.name, ray->tfar);
 		return;
	}

	Hits_XR hit_e;
	hit_e.u = hit->u;
	hit_e.v = hit->v;
	hit_e.prim = hit->primID;
	hit_e.dist = ray->tfar;

	ctxt->hits.push_back(hit_e);
    

	/*
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
	if (U<0)
		U+=T.dwWidth;
	V %= T.dwHeight;	
	if (V<0) 
		V+=T.dwHeight;
	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel		= raw[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
	ctxt->energy *= opac;  
   	*/
}


void RatraceOneRay(RayOptimizedCPU& ray, RayQueryContext& data_hits)
{ 
 	RTCIntersectContext context;
	rtcInitIntersectContext(&context);

	data_hits.context = context;
	data_hits.context.filter =	&FilterIntersectionOne; 
	data_hits.context.flags  =   RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	RTCRayHit rayhit;
	SetRay1(&ray, rayhit);
	rtcIntersect1(IntelScene, &data_hits.context, &rayhit);    
}

#endif 
  
  
float RaytraceEmbreeProcess(CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& N, float range, Face* skip)
{
	/*
	float _u,_v, R;
	bool res = CDB::TestRayTri(P, N, L.tri, _u,_v, R,false);
	if (res) 
	if (range > 0 && range < R) 
 		return 0;
	*/

#ifdef EMBREE_V4
 	RayQueryContext data;
	data.Light = &L;
	data.model = MDL;
	data.skip  = skip;
	data.energy = 1.0f;
	data.count = 0;

			
	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = 0;
	RatraceOneRay(ray, data);

	return data.energy;
#else 
 
 
	RayQueryContext data;
	data.Light = &L;
	data.model = MDL;
	data.skip  = skip;
 	data.count = 0;

	data.energy = 1.0f; 
		
	RayOptimizedCPU ray;
	ray.pos = P;
	ray.dir = N;
	ray.tmax = range;
	ray.tmin = 0;
	RatraceOneRay(ray, data);

	Fvector B;

	if (data.energy == 0)
		return 0;

	float energy = 1.0f;
	for (auto& hit : data.hits)
	{
		// Access to texture
		CDB::TRI* clT = &MDL->get_tris()[hit.prim];	 
		base_Face* F = (base_Face*)(clT->pointer);
		
		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];
		// barycentric coords
		// note: W,U,V order
	
		B.set	(1.0f - hit.u - hit.v, hit.u, hit.v);

		// calc UV
		Fvector2*	cuv = F->getTC0					();
		Fvector2	uv;
		uv.x = cuv[0].x * B.x + cuv[1].x * B.y + cuv[2].x * B.z;
		uv.y = cuv[0].y * B.x + cuv[1].y * B.y + cuv[2].y * B.z;

		int U = iFloor(uv.x*float(T.dwWidth) + .5f);
		int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
		   		
		U %= T.dwWidth;		
		if (U<0)
			U+=T.dwWidth;
		V %= T.dwHeight;	
		if (V<0) 
			V+=T.dwHeight;
		u32* raw = static_cast<u32*>(*T.pSurface);
		u32 pixel		= raw[V*T.dwWidth+U];
		u32 pixel_a		= color_get_A(pixel);
		float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
		energy *= opac;  	
	}
 
	return energy; 	 
#endif

}
		   
/*


struct RayQueryContext8
{
	RTCRayQueryContext context;
	//unsigned int max_next_hits; // maximal number of hits to collect in a single pass
	//unsigned int ray_ID;	
	Hits_XR hits[8][MAX_HITS];
	int count[8];

};


 // 8 Rays
void FilterIntersection8(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext8* ctxt = (RayQueryContext8*)args->context;
 
	for (auto id = 0; id < args->N; id++)
	{
		args->valid[id] = 0;

		RTCHit8* hit = (RTCHit8*) args->hit;  
		RTCRay8* ray = (RTCRay8*) args->ray;

		if (ctxt->count[id] >= MAX_HITS)
			return;

		ctxt->hits[id][ctxt->count[id]].v = hit->v[id];
		ctxt->hits[id][ctxt->count[id]].u = hit->u[id];
		ctxt->hits[id][ctxt->count[id]].prim = hit->primID[id];
		ctxt->hits[id][ctxt->count[id]].dist = ray->tfar[id];

 

		ctxt->count[id] += 1;
	}
}

int HitsTrace8Ray(int* valid, RTCRayHit8& hit8, xr_vector<Hits_XR>& hits)
{
	int count = 8;

	for (int i = 0; i < 8; i++)
	{
		if (hit8.hit.primID[i] == RTC_INVALID_GEOMETRY_ID)
		{
			valid[i] = 0;
 			count--;
		}
		else
		{
			Hits_XR hit; 
			hit.u = hit8.hit.u[i];
			hit.v = hit8.hit.v[i];
			hit.prim = hit8.hit.primID[i];
			hit.dist = hit8.ray.tfar[i];
			hits.push_back(hit);

			ReSetRay8(hit8, i);
		}
	}

	return count;
}

void RayTrace8Ray(int th, xr_vector<RayOptimizedCPU> rays, RayQueryContext8 data)
{
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	data.context = context;

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.filter = &FilterIntersectionOne;
	args.context = &data.context;
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER | RTC_RAY_QUERY_FLAG_INCOHERENT);
 
	struct RTCRayHit8 rayhit;
	
	int valid[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

	int i = 0;
	for (auto ray : rays)
	{
		SetRay8(&ray, rayhit, i);
 		i++;
	}
	rtcIntersect8(valid, IntelScene, &rayhit); //, &args
}
 
void RaytraceEmbreeProcess_Hits8(int th, CDB::MODEL* MDL, xr_vector<RayOptimizedTyped>& rays, xr_vector<float> colors)
{
	xr_vector<RayOptimizedCPU> opt_rays;
	for (auto ray : rays)
		opt_rays.push_back((RayOptimizedCPU) &ray);

	RayQueryContext8 data;
	RayTrace8Ray(th, opt_rays, data);

	for (int i = 0; i < 8; i++)
	{
		if (data.count == 0)
			break;

		float	scale = 1.f;
		Fvector B;
  
		for (u32 I = 0; I < data.count[i]; I++)
		{
			Hits_XR* hit = &data.hits[i][I];

			// Access to texture
			CDB::TRI* clT = &MDL->get_tris()[hit->prim];
			base_Face* F = (base_Face*)(clT->pointer);
			if (0 == F)											continue;
			if (rays[i].skip == F)								continue;

			const Shader_xrLC& SH = F->Shader();
			if (!SH.flags.bLIGHT_CastShadow)					continue;

			if (F->flags.bOpaque)
			{
				// Opaque poly - cache it
				rays[i].Light->tri[0].set(MDL->get_verts()[clT->verts[0]]);
				rays[i].Light->tri[1].set(MDL->get_verts()[clT->verts[1]]);
				rays[i].Light->tri[2].set(MDL->get_verts()[clT->verts[2]]);
				break;
			}

			b_material& M = inlc_global_data()->materials()[F->dwMaterial];
			b_texture& T = inlc_global_data()->textures()[M.surfidx];

			if (T.pSurface.Empty())
			{
				F->flags.bOpaque = true;
				clMsg("* ERROR: RAY-TRACE: Strange face detected... Has alpha without texture...");
				break;
			}

			// barycentric coords
			// note: W,U,V order
			B.set(1.0f - hit->u - hit->v, hit->u, hit->v);

			// calc UV
			Fvector2* cuv = F->getTC0();
			Fvector2  uv;
			uv.x = cuv[0].x * B.x + cuv[1].x * B.y + cuv[2].x * B.z;
			uv.y = cuv[0].y * B.x + cuv[1].y * B.y + cuv[2].y * B.z;

			int U = iFloor(uv.x * float(T.dwWidth) + .5f);
			int V = iFloor(uv.y * float(T.dwHeight) + .5f);

			U %= T.dwWidth;
			if (U < 0)
				U += T.dwWidth;

			V %= T.dwHeight;
			if (V < 0)
				V += T.dwHeight;
			u32* raw = static_cast<u32*>(*T.pSurface);
			u32 pixel = raw[V * T.dwWidth + U];
			u32 pixel_a = color_get_A(pixel);
			float opac = 1.f - _sqr(float(pixel_a) / 255.f);
			colors[i] = scale *= opac;
		}

	}
 
}
 
 */
 
 

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	Msg("error %d: %s", error, str);
}

// OFF PACKED PROCESSING

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
	
	// Создание сцены и добавление геометрии
	 
	struct VertexEmbree { float x, y, z; };
	VertexEmbree* vertices;
	struct TriEmbree { uint32_t point1, point2, point3; };
	TriEmbree* triangles;
	// Добавление вершин

	// Scene
	IntelScene = rtcNewScene(device); 
	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

	vertices = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), inlc_global_data()->RCAST_Model()->get_verts_count());
	triangles = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), inlc_global_data()->RCAST_Model()->get_tris_count());
 	Fbox RQBox;
	
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_verts_count(); i++)
	{
		Fvector verts = inlc_global_data()->RCAST_Model()->get_verts()[i];
		vertices[i] = VertexEmbree{ verts.x, verts.y, verts.z };

		RQBox.modify(verts);
 	}
 
	Fbox RQbox;
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
		CDB::TRI tri = inlc_global_data()->RCAST_Model()->get_tris()[i];
		triangles[i] = TriEmbree{ tri.verts[0], tri.verts[1], tri.verts[2] };
	}
				 
	// Check need
	rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_REFIT);

	rtcCommitGeometry(IntelGeometry);
	rtcAttachGeometry(IntelScene, IntelGeometry);
	
	// Check need
	//rtcSetSceneFlags(IntelScene, RTC_SCENE_FLAG_ROBUST);
	rtcCommitScene(IntelScene);

	RTCBounds bounds;
	rtcGetSceneBounds(IntelScene, &bounds );
	
	

	Msg("SceneBounds: [%f][%f][%f] max [%f][%f][%f] a0: %f, a1: %f", bounds.lower_x, bounds.lower_y, bounds.lower_z, bounds.upper_x, bounds.upper_y, bounds.upper_z, bounds.align0, bounds.align1);
	
	Msg("RcastModel: [%f][%f][%f] max [%f][%f][%f]", VPUSH(RQBox.min) , VPUSH(RQBox.max) ) ;

	// Устанавливать обезательно иле будет в Колбеке PrimID	= 0 
	rtcSetGeometryIntersectFilterFunction(IntelGeometry, &FilterIntersectionOne);	


	for (auto mat : inlc_global_data()->materials())
	{					
		auto shader = inlc_global_data()->shaders().Get(mat.shader);
		auto texture = inlc_global_data()->textures()[mat.surfidx];
		Msg_IN_FILE("Material: %d, name: %s, shader_xrlc: %d, texture: %s, texture_empty: %d (TO Skip LM)",
			mat.surfidx, shader->Name, mat.shader_xrlc, 
			texture.name, texture.pSurface.Empty()
		);


	}

}

void IntelEmbereUNLOAD()
{
	rtcReleaseGeometry(IntelGeometry);
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}

// END INTEL CODE