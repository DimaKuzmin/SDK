#include "stdafx.h"

#include "xrFaceDefs.h"
#include "xrFace.h"
#include "xrHardwareLight.h"
#include "xrDeflector.h"
#include "xrLC_GlobalData.h"
#include "light_point.h"
#include "base_color.h"

#include "cuda_runtime.h"
//#include "cuPrintf.cu"



extern "C"
{
	cudaError_t RunTriCollide(Model* model , RayRequest* ray, RcastResult* hit);
}
  


void xrHardwareLight::LoadRaycastModel(CDB::MODEL* RaycastModel, xr_vector<RayRequest>& InRays)
{
 
}

inline void CheckCudaErr(cudaError_t error);
 
void HemiLight(xr_vector<RayRequest*>& rq, u32 flags, base_lighting& lights, Fvector& P, Fvector& N)
{
	Fvector Ldir;

	if (0 == (flags & LP_dont_hemi))
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
				
				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
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
				if (D <= 0) continue;

				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);

			}

		}
	}
}

void SunLight(xr_vector<RayRequest*>& rq, u32 flags, base_lighting& lights,  Fvector& P, Fvector& N)
{
	Fvector Ldir;

	if (0 == (flags & LP_dont_sun))
	{
 		R_Light* L = &*(lights.sun.begin()), * E = &*(lights.sun.end());
		for (; L != E; L++)
		{
			if (L->type == LT_DIRECT) {
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0) continue;

				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
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
				
				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
			}
		}
	}
}

void RGBLight(xr_vector<RayRequest*>& rq, u32 flags, base_lighting& lights, Fvector& P, Fvector& N)
{
	Fvector Ldir;
	if (0 == (flags & LP_dont_rgb))
	{
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
 
				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
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

				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
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

				RayRequest r;
				r.Position = P;
				r.Normal = Ldir;
				rq.push_back(&r);
			}
			break;
			}
		}
	}
}


void Sort(Fvector& pos, Fvector& dir, u32 flags, xr_vector<RayRequest*>& sorted) //xr_vector<RayRequest>& rq, xr_vector<RayRequest>& out
{
	base_lighting& base = inlc_global_data()->L_static();

	HemiLight(sorted, flags, base, pos, dir);
	SunLight(sorted, flags, base, pos, dir);
	RGBLight(sorted, flags, base, pos, dir);
}

void TriangleIntersection(RayRequest ray)
{
	int id = 0;
	int result = 0;
	
	CDB::MODEL* m = lc_global_data()->RCAST_Model();

	for (int idx = 0; idx < m->get_tris_count(); idx++)
	{
		id++;
		auto tri = m->get_tris()[idx];
		auto ver = m->get_verts();


		Fvector edge1, edge2, tvec, pvec, qvec;
		float det, inv_det;
		float u, v, range;

		Fvector ray_normal	= { ray.Normal.x, ray.Normal.y, ray.Normal.z };
		Fvector ray_pos		= { ray.Position.x, ray.Position.y, ray.Position.z };

		// find vectors for two edges sharing vert0
		Fvector p0 = ver[tri.verts[0]];
		Fvector p1 = ver[tri.verts[1]];
		Fvector p2 = ver[tri.verts[2]];
	 

		edge1.sub(p1, p0);
		edge2.sub(p2, p0);

		// begin calculating determinant - also used to calculate U parameter
		// if determinant is near zero, ray lies in plane of triangle
		pvec.crossproduct(ray_normal, edge2);
		
		det = edge1.dotproduct(pvec);

		if (det < EPS)
			continue;

		tvec.sub(ray_pos, p0);						// calculate distance from vert0 to ray origin
		u = tvec.dotproduct(pvec);					// calculate U parameter and test bounds


		if (u < 0.f || u > det)
			continue;

		qvec.crossproduct(tvec, edge1);				// prepare to test V parameter
		v = ray_normal.dotproduct(qvec);			// calculate V parameter and test bounds

		if (v < 0.f || u + v > det)
			continue;

		range = edge2.dotproduct(qvec);				// calculate t, scale parameters, ray intersects triangle
		inv_det = 1.0f / det;
		
		range *= inv_det;
		u *= inv_det;
		v *= inv_det;

		result++;
		
		Msg("2 idx[%d], r[%d], u[%f], v[%f], dist[%f]", idx, result, u, v, range);

	//	hits[idx_cuda].result[result].triId = idx;
	//	hits[idx_cuda].result[result].u = u;
	//	hits[idx_cuda].result[result].v = v;
	//	hits[idx_cuda].result[result].Distance = 1000;

		//HitSet(hits, 65535, u, v, range);
		//assert(0, "");
	}
}



void xrHardwareLight::TriFindPos(xr_vector<RayRequest>& InRays)
{
	cudaError_t DebugErr = cudaError_t::cudaSuccess;

 	CDB::MODEL* m = lc_global_data()->RCAST_Model();
	xr_vector<Triangle> tris;

	for (int i = 0; i < m->get_tris_count(); i++)
	{
		int p0 = m->get_tris()[i].verts[0];
		int p1 = m->get_tris()[i].verts[1];
		int p2 = m->get_tris()[i].verts[2];
		
		Fvector& v0 = m->get_verts()[p0];
		Fvector& v1 = m->get_verts()[p1];
		Fvector& v2 = m->get_verts()[p2];

		Triangle t; 
		t.FaceID = i;
		t.p1 = v0;
		t.p2 = v1;
		t.p3 = v2;
		tris.push_back(t);
	}

	//MODEL
	DeviceBuffer<Model> model_gpu(16);
	xr_vector<Model> model;
	for (int i = 0; i < 16; i++)
	{
		DeviceBuffer<Triangle> tris_gpu(tris.size());
		cudaMemcpy(tris_gpu.ptr(), tris.data(), tris_gpu.sizeInBytes(), cudaMemcpyHostToDevice);
		Model m;m.tris = tris_gpu.ptr();m.count = tris.size();model.push_back(m);
	}
	model_gpu.copyToBuffer(model.data(), 16);

	// RAYS 
   
	DeviceBuffer<RayRequest> buffer(InRays.size());
	DebugErr = cudaMemcpy(buffer.ptr(), InRays.data(), buffer.sizeInBytes(), cudaMemcpyHostToDevice);
	CheckCudaErr(DebugErr);
 
	DeviceBuffer<RcastResult> hits(InRays.size());
	DebugErr = cudaMemset(hits.ptr(), 0, hits.sizeInBytes());
	CheckCudaErr(DebugErr);
 
	Msg("Working");
   
	CTimer t; t.Start();
	//GPU
	DebugErr = RunTriCollide(model_gpu.ptr(), buffer.ptr(), hits.ptr());
	CheckCudaErr(DebugErr);
	Msg("TimerGPU: %d", (u32)t.GetElapsed_sec());

	//CPU

	/*
	t.Start();
	int ray_id = 0;
	xr_vector<RayRequest> rays;
	for (auto r : InRays)
	{
		ray_id++;
		if (ray_id > 4096)
			break;
		rays.push_back(r);	
	}

	for (auto ray : rays)
		TriangleIntersection(ray);

	Msg("TimerCPU: %f, size: %d", t.GetElapsed_sec(), rays.size() );
	*/


	xr_vector<RcastResult> vectorHIT;
	vectorHIT.resize(hits.count());

	DebugErr = cudaMemcpy(vectorHIT.data(), hits.ptr(), hits.sizeInBytes(), cudaMemcpyDeviceToHost);
	CheckCudaErr(DebugErr);
  
	int id_root = 0;

	int calculated = 0;
	int res = 0;
 	for (auto hits_host : vectorHIT)
	{
		id_root++;
		if (hits_host.total_TRI_used != 0)
		{
			calculated++;
		}



		for (auto r : hits_host.result)
		if (r.triId != 0)
		{
			res++;;
			Msg("Ray u[%f], v[%f], d[%f]", r.u, r.v, r.Distance);
		}
	} 


	Msg("Calculated: %d, result: %d", calculated, res);
}
