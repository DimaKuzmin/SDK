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

RTCScene IntelScene;
RTCGeometry IntelGeometry;
RTCDevice device;

xr_map<u32, bool> opacity;
xr_map<u32, bool> cast_shadowed;

xrCriticalSection csTH;
int last_id = 0;
  
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
  
int count_rays = 0;
CTimer tRQ;
 
RayOptimizedCPU preview;
int prev_count;
int prev_ticks;

xr_vector<Hits_XR> hits_global;
 
u64 global_rays[8];
u64 global_hits = 0;
u64 global_set_data = 0;

CTimer tRAY[8];
CTimer timers[8];
CTimer DWTIMER;
 
u32 old_DWTIME = 0;

extern u64 results;
extern u64 last_results;

u64 ResEmbree = 0;
u64 last_counts = 0;

void IntelClearTimers()
{
	for (int i = 0; i < 8; i ++)
		global_rays[i] = 0;
	global_hits = 0;
	global_set_data = 0;
	Msg("Results: %llu, Embree: %llu", results, ResEmbree);
	results = 0;
	ResEmbree = 0;
	last_counts = 0;
	last_results = 0;
}

// Сделать потом переключалку

// 1 - 8 - 32 - 64 - 96 - 128 - 256 - 512 - 1024
#define MAX_HITS 1024 

struct RayQueryContext
{
	RTCRayQueryContext context;
	//unsigned int max_next_hits; // maximal number of hits to collect in a single pass
	//unsigned int ray_ID;	
	Hits_XR hits[MAX_HITS];
	int count = 0;
	bool last_opactity = false;
	//xr_vector<bool>* opacity;
};

struct RayQueryContext8
{
	RTCRayQueryContext context;
	//unsigned int max_next_hits; // maximal number of hits to collect in a single pass
	//unsigned int ray_ID;	
	Hits_XR hits[8][MAX_HITS];
	int count[8];

};


void FilterIntersectionOne(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*) args->context;
 
	RTCHit* hit = (RTCHit*) args->hit;
	RTCRay* ray = (RTCRay*) args->ray;
 
	if (ctxt->count >= MAX_HITS )	 //|| opacity[hit->primID] 
	{
		//Msg("HITS > %d", MAX_HITS);
		//ctxt->last_opactity = opacity[hit->primID] && !(ctxt->count >= MAX_HITS);
		args->valid[0] = -1;
		return;
	}
	else
	{
		args->valid[0] = 0;
	}

 	ctxt->hits[ctxt->count].v = hit->v;
	ctxt->hits[ctxt->count].u = hit->u;
	ctxt->hits[ctxt->count].prim = hit->primID;
	ctxt->hits[ctxt->count].dist = ray->tfar;
	ctxt->count++;
}

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


void RatraceOneRay(int th, RayOptimizedCPU& ray, RayQueryContext& data_hits)
{ 
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	data_hits.context = context;
 
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.filter = &FilterIntersectionOne;
	args.context = &data_hits.context;	 
	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER); // | RTC_RAY_QUERY_FLAG_INCOHERENT
	//CYCL
	//args.feature_mask = (RTCFeatureFlags) (RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_INSTANCE | RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS);
 
	struct RTCRayHit rayhit;
	SetRay1(&ray, rayhit);
    
	rtcIntersect1(IntelScene, &rayhit, &args);

	/*
	if (MAX_HITS > data_hits.count && data_hits.last_opactity)
	{
		//Msg("data_hits.count:%d, dist: %f, u: %f, v: %f, prim: %d", 
		//	data_hits.count, rayhit.ray.tfar, rayhit.hit.u, rayhit.hit.v, rayhit.hit.primID);
		data_hits.hits[data_hits.count].dist = rayhit.ray.tfar;
		data_hits.hits[data_hits.count].u = rayhit.hit.u;
		data_hits.hits[data_hits.count].v = rayhit.hit.v;
		data_hits.hits[data_hits.count].prim = rayhit.hit.primID;
		data_hits.count++;
	}
	*/

	//global_rays[th] += tRAY[th].GetElapsed_ticks();

	//if (data_hits.count > 150)
	//	Msg_IN_FILE("Ray: %d, cnt: %d", tRAY[th].GetElapsed_ticks(), data_hits.count);

	/*
 	while (rayhit.hit.primID != RTC_INVALID_GEOMETRY_ID)
	{
 		if (rayhit.hit.u != 0 || rayhit.hit.v != 0)
		{
			Hits_XR hit;
			hit.dist = rayhit.ray.tfar;
			hit.u = rayhit.hit.u;
			hit.v = rayhit.hit.v;
			hit.prim = rayhit.hit.primID;
 			hits.push_back(hit);
		}

		SetRay1Hit(rayhit, ray.tmax);
		rtcIntersect1(IntelScene, &rayhit);	  //, &argument_intersect[th]
	}
	*/
    
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
  
int count_HITS_PROCESSED = 0;



float RaytraceEmbreeProcess(int th, CDB::MODEL* MDL, R_Light& L, RayOptimizedCPU& ray, Face* skip)
{
 	if (last_counts + 10000000 < ResEmbree)
	{
		last_counts = ResEmbree;
		Msg("ResultsIntel: %u mln", u64(ResEmbree / 1000000) );
	}
 
	RayQueryContext data;
	RatraceOneRay(th, ray, data);
 
	ResEmbree+=data.count;

	if (data.count == 0)
		return 1;
 
	float	scale = 1.f;
	Fvector B;
	for (u32 I = 0; I < data.count; I++)
	{
		Hits_XR* hit = &data.hits[I];
 
		// Access to texture
		CDB::TRI* clT = &MDL->get_tris()[hit->prim];
		base_Face* F = (base_Face*)(clT->pointer);
				
		if (0 == F)											continue;	
		if (skip == F)										continue;

		const Shader_xrLC& SH = F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)					continue;
 
		if (F->flags.bOpaque)
		{
			// Opaque poly - cache it
			L.tri[0].set(MDL->get_verts()[clT->verts[0]]);
			L.tri[1].set(MDL->get_verts()[clT->verts[1]]);
			L.tri[2].set(MDL->get_verts()[clT->verts[2]]);
			return 0;
		}
 		
		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];

		if (T.pSurface.Empty())
		{
			F->flags.bOpaque = true;
			clMsg("* ERROR: RAY-TRACE: Strange face detected... Has alpha without texture...");
			return 0;
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

		scale *= opac;

		count_HITS_PROCESSED++;
	}
 
	return scale;
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
 
void RaysToHemiLight_Deflector(int th, Fvector& P, Fvector& N, base_color_c& C, base_lighting& lights, Face* skip)
{
	Fvector		Ldir, Pnew;
	Pnew.mad(P, N, 0.01f);
	
	{
		R_Light* L = &*lights.hemi.begin(), * E = &*lights.hemi.end();

		for (; L != E; L++)
		{
			if (L->type == LT_DIRECT)
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0) continue;

				// Trace Light
				Fvector		PMoved;
				PMoved.mad(Pnew, Ldir, 0.001f);

				RayOptimizedCPU ray;
				ray.pos = PMoved;
				ray.dir = Ldir;
				ray.tmax = 1000.f;
				ray.tmin = 0.f;
 
				float scale = L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
				C.hemi += scale;
				
			}
			else
			{
				Fvector pos = L->position;

				// Distance
				float sqD = P.distance_to_sqr(pos);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, P);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0) continue;


				// Trace Light
				float Range = _sqrt(sqD);
				RayOptimizedCPU ray;
				ray.pos = Pnew;
				ray.dir = Ldir;
				ray.tmax = Range;
				ray.tmin = 0.f;
 
				float scale = L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
				float A = scale / (L->attenuation0 + L->attenuation1 * Range + L->attenuation2 * sqD);
				C.hemi += A;
			}
		}
	}
}

void RaysToRGBLight_Deflector(int th, Fvector& P, Fvector& N, base_color_c& C, base_lighting& lights, Face* skip)
{
	Fvector		Ldir, Pnew;
	Pnew.mad(P, N, 0.01f);
	
	R_Light* L = &*lights.rgb.begin(), * E = &*lights.rgb.end();
	for (; L != E; L++)
	{
		switch (L->type)
		{
			case LT_DIRECT:
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0) continue;
 
				RayOptimizedCPU ray;
				ray.pos = Pnew;
				ray.dir = Ldir;
				ray.tmax = 1000.0f;
				ray.tmin = 0.f;

				float scale = D * L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
				C.rgb.x += scale * L->diffuse.x;
				C.rgb.y += scale * L->diffuse.y;
				C.rgb.z += scale * L->diffuse.z;

				//Msg("RGB COLOR");
			}
			break;
			case LT_POINT:
			{
				// Distance
				float sqD = P.distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, P);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0)			continue;
			    
				// Trace Light
				float Range = _sqrt(sqD);
				RayOptimizedCPU ray;
				ray.pos = Pnew;
				ray.dir = Ldir;
				ray.tmax = Range;
				ray.tmin = 0.f;
 				 
				float scale = D * L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
				float A;
				if (inlc_global_data()->gl_linear())
					A = 1 - Range / L->range;
				else
					A = scale * (1 / (L->attenuation0 + L->attenuation1 * Range + L->attenuation2 * sqD) - Range * L->falloff);
			
				C.rgb.x += A * L->diffuse.x;
				C.rgb.y += A * L->diffuse.y;
				C.rgb.z += A * L->diffuse.z;

				//Msg("RGB COLOR");
			}
			break;
			case LT_SECONDARY:
			{
				// Distance
				float sqD = P.distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, P);
				Ldir.normalize_safe();
				float	D = Ldir.dotproduct(N);
				if (D <= 0) continue;
				D *= -Ldir.dotproduct(L->direction);
				if (D <= 0) continue;
	
				// Jitter + trace light -> monte-carlo method
 				Fvector	Psave = L->position, Pdir;
				L->position.mad(Pdir.random_dir(L->direction, PI_DIV_4), .05f);
				float R = _sqrt(sqD);
				RayOptimizedCPU ray;
				ray.pos = Pnew;
				ray.dir = Ldir;
				ray.tmax = R;
				ray.tmin = 0.f;

 
				float scale = powf(D, 1.f / 8.f) * L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
				float A = scale * (1 - R / L->range);
				L->position = Psave;

				C.rgb.x += A * L->diffuse.x;
				C.rgb.y += A * L->diffuse.y;
				C.rgb.z += A * L->diffuse.z;

				//Msg("RGB COLOR");
			}
			break;
		}
	}
}

void RaysToSUNLight_Deflector(int th, Fvector& P, Fvector& N, base_color_c& C, base_lighting& lights, Face* skip)
{
	Fvector		Ldir, Pnew;
	Pnew.mad(P, N, 0.01f);
	R_Light* L = &*lights.sun.begin(), * E = &*lights.sun.end();

	for (; L != E; L++)
	{
		if (L->type == LT_DIRECT)
		{
			// Cos
			Ldir.invert(L->direction);
			float D = Ldir.dotproduct(N);
			if (D <= 0) continue;

			// Trace Light
 			RayOptimizedCPU ray;
			ray.pos = Pnew;
			ray.dir = Ldir;
			ray.tmax = 1000.f;
			ray.tmin = 0.f;
			
			float scale = L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip);
			C.sun += scale;
			//rays.push_back(r);
		}
		else
		{
			// Distance
			float sqD = P.distance_to_sqr(L->position);
			if (sqD > L->range2) continue;

			// Dir
			Ldir.sub(L->position, P);
			Ldir.normalize_safe();
			float D = Ldir.dotproduct(N);
			if (D <= 0)			continue;

		
			// Trace Light			
			float Range = _sqrt(sqD);
			RayOptimizedCPU ray;
			ray.pos = Pnew;
			ray.dir = Ldir;
			ray.tmax = Range;
			ray.tmin = 0.f;
			
			float scale = D * L->energy * RaytraceEmbreeProcess(th, inlc_global_data()->RCAST_Model(), *L, ray, skip) ;
			float A = scale / (L->attenuation0 + L->attenuation1 * Range + L->attenuation2 * sqD);
			C.sun += A;
			
			//rays.push_back(r);

		}
	}
}

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	Msg("error %d: %s", error, str);
}

// OFF PACKED PROCESSING

void IntelEmbereLOAD()
{
	std::string config;
	if (strstr(Core.Params, "-use_avx"))
		config = "threads=8,isa=avx2";
	else if (strstr(Core.Params, "-use_sse"))
		config = "threads=8,isa=sse4.2";
	else
		config = "threads=8,isa=sse2";

	if (strstr(Core.Params, "-use_avx"))
		Msg("Initilized AVX");
	else
		Msg("Initilized NO INSTRUCTION");


	device = rtcNewDevice(config.c_str());
	if (!device)
		Msg("Cant Create Device !!!");

	rtcSetDeviceErrorFunction(device, errorFunction, NULL);

	Msg("Packed4: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED));
	Msg("Packed8: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED));
	Msg("Packed16: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED));

	// Создание сцены и добавление геометрии
	 
	struct VertexEmbree { float x, y, z; };
	VertexEmbree* vertices;
	struct TriEmbree { uint32_t point1, point2, point3; };
	TriEmbree* triangles;
	// Добавление вершин
	 
	Msg("DEVICE CULLING: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED) );
	Msg("DEVICE IGNORE FACES: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED));
	Msg("DEVICE COMPACT FACES: %d", rtcGetDeviceProperty(device, RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED));



	IntelScene = rtcNewScene(device);
	 
	//rtcSetSceneFlags(IntelScene, RTC_SCENE_FLAG_COMPACT);
	//rtcSetSceneFlags(IntelScene, RTC_SCENE_FLAG_ROBUST);
	//rtcSetSceneBuildQuality(IntelScene, RTC_BUILD_QUALITY_HIGH);
 
	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

	vertices = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), inlc_global_data()->RCAST_Model()->get_verts_count());
	triangles = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), inlc_global_data()->RCAST_Model()->get_tris_count());
 
	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_verts_count(); i++)
	{
		Fvector verts = inlc_global_data()->RCAST_Model()->get_verts()[i];
		vertices[i] = VertexEmbree{ verts.x, verts.y, verts.z };
	}

	int TriOpaque = 0;
	int TriNotOpque = 0;

	int TriShadowed = 0;
	int TriNotShadowed = 0;

	for (int i = 0; i < inlc_global_data()->RCAST_Model()->get_tris_count(); i++)
	{
		CDB::TRI tri = inlc_global_data()->RCAST_Model()->get_tris()[i];
		base_Face* F = (base_Face*)(tri.pointer);
		const Shader_xrLC& SH = F->Shader();

		opacity[i] = F->flags.bOpaque;
		cast_shadowed[i] = SH.flags.bLIGHT_CastShadow;
		if (opacity[i])
			TriOpaque++;
		else
			TriNotOpque++;

		if (cast_shadowed[i])
			TriShadowed++;
		else
			TriNotShadowed++;

		triangles[i] = TriEmbree{ tri.verts[0], tri.verts[1], tri.verts[2] };
	}

	Msg("Opacue: true: %d / false: %d", TriOpaque, TriNotOpque);
	Msg("Shadowed: true: %d / false: %d", TriShadowed, TriNotShadowed);
		
	rtcCommitGeometry(IntelGeometry);
	rtcAttachGeometry(IntelScene, IntelGeometry);
	

	rtcCommitScene(IntelScene);

	// Устанавливать обезательно иле будет в Колбеке PrimID	= 0 
	rtcSetGeometryIntersectFilterFunction(IntelGeometry, &FilterIntersectionOne);		
 
	// Только для USER GEOM
	//argument_intersect.intersect = &IntersectFunction;
	//rtcSetGeometryIntersectFunction(geometry, &IntersectFunction);
}

void IntelEmbereUNLOAD()
{
	for (auto i = 0; i < 8; i++)
	{
		rtcReleaseGeometry(IntelGeometry);
		rtcReleaseScene(IntelScene);
	}

	rtcReleaseDevice(device);
}

// END INTEL CODE