#include "stdafx.h"
#pragma hdrstop
#pragma warning(push)
#pragma warning(disable:4995)
#include <xmmintrin.h>	 
#include <immintrin.h>
#include <emmintrin.h>
#pragma warning(pop)

#include "xrCDB.h"
#include <directx\d3dx9.h>

using namespace		CDB;
using namespace		Opcode;

// can you say "barebone"?
#ifndef _MM_ALIGN16
#	define _MM_ALIGN16	__declspec(align(16))
#endif // _MM_ALIGN16

struct	_MM_ALIGN16		vec_t : public Fvector3
{
	float		pad;
};
//static vec_t	vec_c	( float _x, float _y, float _z)	{ vec_t v; v.x=_x;v.y=_y;v.z=_z;v.pad=0; return v; }

struct _MM_ALIGN16		aabb_t {
	vec_t		min;
	vec_t		max;
};
struct _MM_ALIGN16		ray_t {
	vec_t		pos;
	vec_t		inv_dir;
	vec_t		fwd_dir;
};

struct _MM_ALIGN16 RayOptimized
{
	ICF void set(Fvector3 o, Fvector3 d)
	{
		origin = o;
		direction = d;
		inv_direction.set(1 / d.x, 1 / d.y, 1 / d.z);
		sign[0] = (inv_direction.x < 0) ? 1 : 0;
		sign[1] = (inv_direction.y < 0) ? 1 : 0;
		sign[2] = (inv_direction.z < 0) ? 1 : 0;
	}
	Fvector3 origin;
	Fvector3 direction;
	Fvector3 inv_direction;
	int sign[3];
};

struct _MM_ALIGN16 BOX_Optimized
{
public:
	ICF BOX_Optimized(const Fvector3& min, const Fvector3& max)
	{
		//RAssert(min < max);
		bounds[0] = min;
		bounds[1] = max;
	}
 	Fvector3 bounds[2];

};

ICF static bool intersect_v1(const BOX_Optimized& b, const RayOptimized& r, float t0, float t1)
{
	 
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	tmin = (b.bounds[r.sign[0]].x - r.origin.x) * r.inv_direction.x;
	tmax = (b.bounds[1 - r.sign[0]].x - r.origin.x) * r.inv_direction.x;
	tymin = (b.bounds[r.sign[1]].y - r.origin.y) * r.inv_direction.y;
	tymax = (b.bounds[1 - r.sign[1]].y - r.origin.y) * r.inv_direction.y;
	
	if ((tmin > tymax) || (tymin > tmax))
		return false;
	
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	
	tzmin = (b.bounds[r.sign[2]].z - r.origin.z) * r.inv_direction.z;
	tzmax = (b.bounds[1 - r.sign[2]].z - r.origin.z) * r.inv_direction.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	bool val = (tmin < t1) && (tmax > t0);

	//if (val)
	//Msg("tmin[%f] tmax[%f], max_dist[%f]",tmin,tmax, t0);

	return (val);
}

struct _MM_ALIGN16 RayOptimized_v2
{
	float origin[3];
	float dir[3];
	float dir_inv[3];
  
	void SetOrigin(Fvector v)
	{
		origin[0] = v.x;
		origin[1] = v.y;
		origin[2] = v.z;
	}

	void SetDir(Fvector v)
	{
		dir[0] = v.x;
		dir[1] = v.y;
		dir[2] = v.z;
	}

	void SetInvDir(Fvector v)
	{
		dir_inv[0] = v.x;
		dir_inv[1] = v.y;
		dir_inv[2] = v.z;
	}
};

/// An axis-aligned bounding box.
struct _MM_ALIGN16 BOX_Optimized_v2
{
	float min[3];
	float max[3];
};

ICF static inline float min(float x, float y) {
	return x < y ? x : y;
}

ICF static inline float max(float x, float y)
{
	return x > y ? x : y;
}


ICF static bool intersection_v2(RayOptimized_v2* ray, BOX_Optimized_v2* box, float max_dist)
{
	float tmin = -max_dist, tmax = max_dist;
	for (int i = 0; i < 3; ++i)
	{
		float t1 = (box->min[i] - ray->origin[i]) * ray->dir_inv[i];
		float t2 = (box->max[i] - ray->origin[i]) * ray->dir_inv[i];

		tmin = max(tmin, min(t1, t2));
		tmax = min(tmax, max(t1, t2));
	}
	return tmin < tmax;
}





struct ray_segment_t {
	float		t_near, t_far;
};

ICF u32& uf(float& x) { return (u32&)x; }

ICF BOOL check_min(const ray_t& ray, Fvector& MaxT, const Fvector& min, const Fvector& max, Fvector& coord)
{
	bool Inside = true;

	if (ray.pos[0] < min[0])
	{
		coord[0] = min[0];
		Inside = FALSE;
		if (uf(ray.inv_dir[0]))	MaxT[0] = (min[0] - ray.pos[0]) * ray.inv_dir[0]; // Calculate T distances to candidate planes
	}

	if (ray.pos[1] < min[1])
	{
		coord[1] = min[1];
		Inside = FALSE;
		if (uf(ray.inv_dir[1]))	MaxT[1] = (min[1] - ray.pos[1]) * ray.inv_dir[1]; // Calculate T distances to candidate planes
	}


	if (ray.pos[2] < min[2])
	{
		coord[2] = min[2];
		Inside = FALSE;
		if (uf(ray.inv_dir[2]))	MaxT[2] = (min[2] - ray.pos[2]) * ray.inv_dir[2]; // Calculate T distances to candidate planes
	}

	return Inside;
}

ICF BOOL check_max(const ray_t& ray, Fvector& MaxT, const Fvector& min, const Fvector& max, Fvector& coord)
{
	bool Inside = true;

	if (ray.pos[0] > max[0])
	{
		coord[0] = max[0];
		Inside = FALSE;
		if (uf(ray.inv_dir[0]))	MaxT[0] = (max[0] - ray.pos[0]) * ray.inv_dir[0]; // Calculate T distances to candidate planes
	}

	if (ray.pos[1] > max[1])
	{
		coord[1] = max[1];
		Inside = FALSE;
		if (uf(ray.inv_dir[1]))	MaxT[1] = (max[1] - ray.pos[1]) * ray.inv_dir[1]; // Calculate T distances to candidate planes
	}

	if (ray.pos[2] > max[2])
	{
		coord[2] = max[2];
		Inside = FALSE;
		if (uf(ray.inv_dir[2]))	MaxT[2] = (max[2] - ray.pos[2]) * ray.inv_dir[2]; // Calculate T distances to candidate planes
	}

	return Inside;
}

xrCriticalSection csCDB;

ICF BOOL	isect_fpu_t(const Fvector& min, const Fvector& max, const ray_t& ray, Fvector& coord)
{
	Fvector				MaxT;
	MaxT.x = MaxT.y = MaxT.z = -1.0f;
	//BOOL Inside = TRUE;
 
	// Find candidate planes.
	/*
	if (ray.pos[0] < min[0])
	{
		coord[0] = min[0];
		Inside = FALSE;
		if (uf(ray.inv_dir[0]))	MaxT[0] = (min[0] - ray.pos[0]) * ray.inv_dir[0]; // Calculate T distances to candidate planes
	}
	else
	if (ray.pos[0] > max[0] && use_max)
	{
		coord[0] = max[0];
		Inside = FALSE;
		if (uf(ray.inv_dir[0]))	MaxT[0] = (max[0] - ray.pos[0]) * ray.inv_dir[0]; // Calculate T distances to candidate planes
	}

	if (ray.pos[1] < min[1])
	{
		coord[1] = min[1];
		Inside = FALSE;
		if (uf(ray.inv_dir[1]))	MaxT[1] = (min[1] - ray.pos[1]) * ray.inv_dir[1]; // Calculate T distances to candidate planes
	}
	else
	if (ray.pos[1] > max[1] && use_max)
	{
		coord[1] = max[1];
		Inside = FALSE;
		if (uf(ray.inv_dir[1]))	MaxT[1] = (max[1] - ray.pos[1]) * ray.inv_dir[1]; // Calculate T distances to candidate planes
	}

	if (ray.pos[2] < min[2]) 
	{
		coord[2] = min[2];
		Inside = FALSE;
		if (uf(ray.inv_dir[2]))	MaxT[2] = (min[2] - ray.pos[2]) * ray.inv_dir[2]; // Calculate T distances to candidate planes
	}
	else 
	if (ray.pos[2] > max[2] && use_max)
	{
		coord[2] = max[2];
		Inside = FALSE;
		if (uf(ray.inv_dir[2]))	MaxT[2] = (max[2] - ray.pos[2]) * ray.inv_dir[2]; // Calculate T distances to candidate planes
	}
	*/


	//bool checked = check_min(ray, MaxT, min, max, coord) && check_max(ray, MaxT, min, max, coord);
	bool c_min = check_min(ray, MaxT, min, max, coord);
	bool c_max = check_max(ray, MaxT, min, max, coord);

	// Ray ray.pos inside bounding box
	if (c_min && c_max)
	{
		coord = ray.pos;
		return		true;
	}

	// Get largest of the maxT's for final choice of intersection
	u32 WhichPlane = 0;
	if (MaxT[1] > MaxT[0])			WhichPlane = 1;
	if (MaxT[2] > MaxT[WhichPlane])	WhichPlane = 2;

	// Check final candidate actually inside box (if max < 0)
	if (uf(MaxT[WhichPlane]) & 0x80000000) return false;

	if (0 == WhichPlane) 
	{	// 1 & 2
		coord[1] = ray.pos[1] + MaxT[0] * ray.fwd_dir[1];
		if ((coord[1] < min[1]) || (coord[1] > max[1]))	return false;
		coord[2] = ray.pos[2] + MaxT[0] * ray.fwd_dir[2];
		if ((coord[2] < min[2]) || (coord[2] > max[2]))	return false;
		return true;
	}
	
	if (1 == WhichPlane) {	// 0 & 2
		coord[0] = ray.pos[0] + MaxT[1] * ray.fwd_dir[0];
		if ((coord[0] < min[0]) || (coord[0] > max[0]))	return false;
		coord[2] = ray.pos[2] + MaxT[1] * ray.fwd_dir[2];
		if ((coord[2] < min[2]) || (coord[2] > max[2]))	return false;
		return true;
	}
	
	if (2 == WhichPlane) {	// 0 & 1
		coord[0] = ray.pos[0] + MaxT[2] * ray.fwd_dir[0];
		if ((coord[0] < min[0]) || (coord[0] > max[0]))	return false;
		coord[1] = ray.pos[1] + MaxT[2] * ray.fwd_dir[1];
		if ((coord[1] < min[1]) || (coord[1] > max[1]))	return false;
		return true;
	}
	return false;
}

// turn those verbose intrinsics into something readable.
#define loadps(mem)			_mm_load_ps((const float * const)(mem))
#define storess(ss,mem)		_mm_store_ss((float * const)(mem),(ss))
#define minss				_mm_min_ss
#define maxss				_mm_max_ss
#define minps				_mm_min_ps
#define maxps				_mm_max_ps
#define mulps				_mm_mul_ps
#define subps				_mm_sub_ps
#define rotatelps(ps)		_mm_shuffle_ps((ps),(ps), 0x39)	// a,b,c,d -> b,c,d,a
#define muxhps(low,high)	_mm_movehl_ps((low),(high))		// low{a,b,c,d}|high{e,f,g,h} = {c,d,g,h}

static const float flt_plus_inf = -logf(0);	// let's keep C and C++ compilers happy.
static const float _MM_ALIGN16
ps_cst_plus_inf[4] = { flt_plus_inf,  flt_plus_inf,  flt_plus_inf,  flt_plus_inf },
ps_cst_minus_inf[4] = { -flt_plus_inf, -flt_plus_inf, -flt_plus_inf, -flt_plus_inf };

ICF BOOL isect_sse_t(const aabb_t& box, const ray_t& ray, float& dist) {
	// you may already have those values hanging around somewhere
	const __m128
		plus_inf = loadps(ps_cst_plus_inf),
		minus_inf = loadps(ps_cst_minus_inf);

	// use whatever's apropriate to load.
	const __m128
		box_min = loadps(&box.min),
		box_max = loadps(&box.max),
		pos = loadps(&ray.pos),
		inv_dir = loadps(&ray.inv_dir);

	// use a div if inverted directions aren't available
	const __m128 l1 = mulps(subps(box_min, pos), inv_dir);
	const __m128 l2 = mulps(subps(box_max, pos), inv_dir);

	// the order we use for those min/max is vital to filter out
	// NaNs that happens when an inv_dir is +/- inf and
	// (box_min - pos) is 0. inf * 0 = NaN
	const __m128 filtered_l1a = minps(l1, plus_inf);
	const __m128 filtered_l2a = minps(l2, plus_inf);

	const __m128 filtered_l1b = maxps(l1, minus_inf);
	const __m128 filtered_l2b = maxps(l2, minus_inf);

	// now that we're back on our feet, test those slabs.
	__m128 lmax = maxps(filtered_l1a, filtered_l2a);
	__m128 lmin = minps(filtered_l1b, filtered_l2b);

	// unfold back. try to hide the latency of the shufps & co.
 	const __m128 lmax0 = rotatelps(lmax);
	const __m128 lmin0 = rotatelps(lmin);
	lmax = minss(lmax, lmax0);
	lmin = maxss(lmin, lmin0);

	const __m128 lmax1 = muxhps(lmax, lmax);
	const __m128 lmin1 = muxhps(lmin, lmin);
	lmax = minss(lmax, lmax1);
	lmin = maxss(lmin, lmin1);

	const BOOL ret = _mm_comige_ss(lmax, _mm_setzero_ps()) & _mm_comige_ss(lmax, lmin);

	storess(lmin, &dist);
	//storess	(lmax, &rs.t_far);
	
	return  ret;
}
 
template <bool bCull, bool bFirst, bool bNearest>
class _MM_ALIGN16	ray_collider
{
public:
	COLLIDER* dest;
	TRI* tris;
	Fvector* verts;

	ray_t			ray;
	RayOptimized	ray_optimize;
	RayOptimized_v2 ray_optimize_v2;

 
	float			rRange;
	float			rRange2;
	bool			bUseSSE;

	IC void			_init(COLLIDER* CL, Fvector* V, TRI* T, const Fvector& C, const Fvector& D, float R)
	{
		dest = CL;
		tris = T;
		verts = V;
		
		ray.pos.set(C);
		ray.inv_dir.set(1.f, 1.f, 1.f).div(D);
		ray.fwd_dir.set(D);
		
		ray_optimize.set(C, D);
		ray_optimize_v2.SetDir(D);
		ray_optimize_v2.SetOrigin(C);
		ray_optimize_v2.SetInvDir(ray.inv_dir);


		rRange = R;
		rRange2 = R * R;
 
		if (!bUseSSE)
		{
			// for FPU - zero out inf
			if (_abs(D.x) > flt_eps) {}
			else ray.inv_dir.x = 0;
			if (_abs(D.y) > flt_eps) {}
			else ray.inv_dir.y = 0;
			if (_abs(D.z) > flt_eps) {}
			else ray.inv_dir.z = 0;
		}
	}

	// fpu
	ICF BOOL		_box_fpu(const Fvector& bCenter, const Fvector& bExtents, Fvector& coord)
	{
		Fbox		BB;
		BB.min.sub(bCenter, bExtents);
		BB.max.add(bCenter, bExtents);
		return 		isect_fpu_t(BB.min, BB.max, ray, coord);
	}

	// sse
	ICF BOOL		_box_sse(const Fvector& bCenter, const Fvector& bExtents, float& dist)
	{
		aabb_t		box;
		/*
			box.min.sub (bCenter,bExtents);	box.min.pad = 0;
			box.max.add	(bCenter,bExtents); box.max.pad = 0;
		*/
		__m128 CN = _mm_unpacklo_ps(_mm_load_ss((float*)&bCenter.x), _mm_load_ss((float*)&bCenter.y));
		CN = _mm_movelh_ps(CN, _mm_load_ss((float*)&bCenter.z));
		__m128 EX = _mm_unpacklo_ps(_mm_load_ss((float*)&bExtents.x), _mm_load_ss((float*)&bExtents.y));
		EX = _mm_movelh_ps(EX, _mm_load_ss((float*)&bExtents.z));

		_mm_store_ps((float*)&box.min, _mm_sub_ps(CN, EX));
		_mm_store_ps((float*)&box.max, _mm_add_ps(CN, EX));
		
		return 		isect_sse_t(box, ray, dist);
	}

	IC bool			_tri(u32* p, float& u, float& v, float& range)
	{
		Fvector edge1, edge2, tvec, pvec, qvec;
		float	det, inv_det;

		// find vectors for two edges sharing vert0
		Fvector& p0 = verts[p[0]];
		Fvector& p1 = verts[p[1]];
		Fvector& p2 = verts[p[2]];
		edge1.sub(p1, p0);
		edge2.sub(p2, p0);
		// begin calculating determinant - also used to calculate U parameter
		// if determinant is near zero, ray lies in plane of triangle
		pvec.crossproduct(ray.fwd_dir, edge2);
		det = edge1.dotproduct(pvec);
		if (bCull)
		{
			if (det < EPS)  return false;
			tvec.sub(ray.pos, p0);						// calculate distance from vert0 to ray origin
			u = tvec.dotproduct(pvec);					// calculate U parameter and test bounds
			if (u < 0.f || u > det) return false;
			qvec.crossproduct(tvec, edge1);				// prepare to test V parameter
			v = ray.fwd_dir.dotproduct(qvec);			// calculate V parameter and test bounds
			if (v < 0.f || u + v > det) return false;
			range = edge2.dotproduct(qvec);				// calculate t, scale parameters, ray intersects triangle
			inv_det = 1.0f / det;
			range *= inv_det;
			u *= inv_det;
			v *= inv_det;
		}
		else
		{
			if (det > -EPS && det < EPS) return false;
			inv_det = 1.0f / det;
			tvec.sub(ray.pos, p0);						// calculate distance from vert0 to ray origin
			u = tvec.dotproduct(pvec) * inv_det;			// calculate U parameter and test bounds
			if (u < 0.0f || u > 1.0f)    return false;
			qvec.crossproduct(tvec, edge1);				// prepare to test V parameter
			v = ray.fwd_dir.dotproduct(qvec) * inv_det;	// calculate V parameter and test bounds
			if (v < 0.0f || u + v > 1.0f) return false;
			range = edge2.dotproduct(qvec) * inv_det;		// calculate t, ray intersects triangle
		}

		return true;
	}

	void			_prim(DWORD prim)
	{
		float	u, v, r;
		if (!_tri(tris[prim].verts, u, v, r))	return;
 		if (r <= 0 || r > rRange)					return;

		if (bNearest)
		{
			if (dest->r_count())
			{
				RESULT& R = *dest->r_begin();
				if (r < R.range) {
					R.id = prim;
					R.range = r;
					R.u = u;
					R.v = v;
					R.verts[0] = verts[tris[prim].verts[0]];
					R.verts[1] = verts[tris[prim].verts[1]];
					R.verts[2] = verts[tris[prim].verts[2]];
					R.dummy = tris[prim].dummy;
					rRange = r;
					rRange2 = r * r;
				}
			}
			else {
				RESULT& R = dest->r_add();
				R.id = prim;
				R.range = r;
				R.u = u;
				R.v = v;
				R.verts[0] = verts[tris[prim].verts[0]];
				R.verts[1] = verts[tris[prim].verts[1]];
				R.verts[2] = verts[tris[prim].verts[2]];
				R.dummy = tris[prim].dummy;
				rRange = r;
				rRange2 = r * r;
			}
		}
		else {
			RESULT& R = dest->r_add();
			R.id = prim;
			R.range = r;
			R.u = u;
			R.v = v;
			R.verts[0] = verts[tris[prim].verts[0]];
			R.verts[1] = verts[tris[prim].verts[1]];
			R.verts[2] = verts[tris[prim].verts[2]];
			R.dummy = tris[prim].dummy;
		}
	}

	//u32 count;

	void			_stab(const AABBNoLeafNode* node)
	{
		//count++;
		// Should help
		_mm_prefetch((char*)node->GetNeg(), _MM_HINT_NTA);

		// Actual ray/aabb test
		
		 
		if (bUseSSE)
		{
			// use SSE
			float		d;
			if (!_box_sse((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents, d))	return;
			//if (!isect_sse_fast(node, ray, d)) return;
			if (d > rRange)																	return;
		}
		else 
		{
			// use FPU
			Fvector		P;
			if (!_box_fpu((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents, P))	return;
			if (P.distance_to_sqr(ray.pos) > rRange2)											return;
		}
		 

		/*
		Fbox		BB;
		BB.min.sub((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents);
		BB.max.add((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents);

	
		
		BOX_Optimized box(BB.min, BB.max);
		if (!intersect_v1(box, ray_optimize, -rRange, rRange))
			return;
		

		BOX_Optimized_v2 bbox;
		bbox.min[0] = BB.min.x;
		bbox.min[1] = BB.min.y;
		bbox.min[2] = BB.min.z;
		bbox.max[0] = BB.max.x;
		bbox.max[1] = BB.max.y;
		bbox.max[2] = BB.max.z;
		if (!intersection_v2(&ray_optimize_v2, &bbox, rRange))
			return;
		*/

/*
		
		//TextBOX   V1
		Fbox		BB;
		BB.min.sub((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents);
		BB.max.add((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents);

//#define use_method_v1

#ifdef use_method_v1
		BOX_Optimized box(BB.min, BB.max);		 
		if (!intersect_v1(box, *r, -rRange, rRange))
			return;
	
#else		 
		//V2
		ad_box bbox;
		bbox.min[0] = BB.min.x;
		bbox.min[1] = BB.min.y;
		bbox.min[2] = BB.min.z;
		bbox.max[0] = BB.max.x;
		bbox.max[1] = BB.max.y;
		bbox.max[2] = BB.max.z;
		if (!intersection_v2(&r_a, &bbox, rRange))
			return;
		 
#endif

*/
		// 1st chield
		if (node->HasLeaf())
			_prim(node->GetPrimitive());
		else	
			_stab(node->GetPos());

		// Early exit for "only first"
		if (bFirst && dest->r_count())														return;

		// 2nd chield
		if (node->HasLeaf2())
			_prim(node->GetPrimitive2());
		else					
			_stab(node->GetNeg());
	}
};

xrCriticalSection csRAY;

u64 IDX = 0;
u64 RES = 0;
u64 CCOUNT = 0;

/*
u64 TIDX = 0;
  

u64 countsTOTAL = 0;
*/

CTimer* ttimer = 0;
float old_sec = 0;

u64 tri_time = 0;
u64 pos_time = 0;
 
u64 total_tri = 0;
u64 total_ms = 0;

bool start_thread = false;
#include <thread>

void StartLog()
{

	for (;;)
	{
		Sleep(5000);
		Msg("TIDX: %u, CCOUNT[%u], RES: %u", IDX / 1000000, CCOUNT / 1000000, RES );
	}
}

void	COLLIDER::ray_query(const MODEL* m_def, const Fvector& r_start, const Fvector& r_dir, float r_range, int INSTR_IDX) //0 = FPU, 1= SSE, 2 = AVX
{
	/*
	if (!start_thread)
	{
		start_thread = true;
		std::thread* th = new std::thread(StartLog);
	}

	IDX++;
	*/
	/*
	csRAY.Enter();
	if (!ttimer)
	{
		ttimer = xr_new<CTimer>();
		ttimer->Start();
 	}
	csRAY.Leave();

	
	CTimer timer; 
	timer.Start();
   	*/
 
	m_def->syncronize();
 	

	// Get nodes
	const AABBNoLeafTree* T = (const AABBNoLeafTree*)m_def->tree->GetTree();
	const AABBNoLeafNode* N = T->GetNodes();
	r_clear();

	bool use_sse = INSTR_IDX == 1 ? true : false;
 
	// SSE
	// Binary dispatcher
	if (ray_mode & OPT_CULL)
	{
		if (ray_mode & OPT_ONLYFIRST) 
		{
			if (ray_mode & OPT_ONLYNEAREST) 
			{
				ray_collider<true, true, true>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
			else {
				ray_collider<true, true, false>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
		}
		else
		{
			if (ray_mode & OPT_ONLYNEAREST)
			{
				ray_collider<true, false, true>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
			else
			{
				ray_collider<true, false, false>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
		}
	}
	else {
		if (ray_mode & OPT_ONLYFIRST) 
		{
			if (ray_mode & OPT_ONLYNEAREST) 
			{
				ray_collider<false, true, true>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
			else 
			{
				ray_collider<false, true, false>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
		}
		else 
		{
			if (ray_mode & OPT_ONLYNEAREST) {
				ray_collider<false, false, true>		RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
				RC._stab(N);
			}
			else
			{
				ray_collider<false, false, false>	RC;
				RC.bUseSSE = use_sse;
				RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
 				
				//CTimer t; t.Start();
				RC._stab(N);
				//RES += RC.dest->r_count();
				//CCOUNT += RC.count;
				//tri_time += t.GetElapsed_ticks();
			}
		}
	}
	
	/*
	TIDX += timer.GetElapsed_ticks();
	
	csRAY.Enter();
	if ( (TIDX / 10000) > 10000)
	{
		if (timer.GetElapsed_ticks() == 0)
			timer.Start();

		Msg("RAY{IDX = %u, calls = %u, Count = %u MILLION, Time = %u}", IDX, countsTOTAL, RES / 1000000, TIDX / 10000);
		RES = 0;
		TIDX = 0;
		countsTOTAL = 0;
	}
	csRAY.Leave();
	*/
 	/*
	csRAY.Enter();
	if (IDX % 1000000 == 0)
	{
		u32 TRI = u32(u64(tri_time * ttimer->GetElapsed_ms()) / 10000000);

		total_tri += TRI;
		total_ms += ttimer->GetElapsed_ms();
		
		Msg("IDX[%u], RES[%u], time[%u], TRI: %u, total[%u], ms[%d]", IDX, RES, ttimer->GetElapsed_ms(), TRI, total_tri, total_ms);

		ttimer->Start();
		tri_time = 0;
		pos_time = 0;
	}
	csRAY.Leave();
 	*/
}

