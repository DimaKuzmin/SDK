#include "stdafx.h"

#include "xrdeflector.h"
#include "cl_intersect.h"
#include "xrlc_globaldata.h"
#include "light_point.h"
#include "xrFace.h"
#include "embree_raytracing/EmbreeRayTrace.h"

// ORIGINAL 
void Jitter_Select(Fvector2*& Jitter, u32& Jcount)
{
	static Fvector2 Jitter1[1] = {
		{0,0}
	};
	static Fvector2 Jitter4[4] = {
		{-1,-1}, {1,-1}, {1,1}, {-1,1}
	};
	static Fvector2 Jitter9[9] = {
		{-1,-1},	{0,-1},		{1,-1},
		{-1,0},		{0,0},		{1,0},
		{-1,1},		{0,1},		{1,1}
	};

	switch (g_params().m_lm_jitter_samples)
	{
	case 1:
		Jcount = 1;
		Jitter = Jitter1;
		break;
	case 9:
		Jcount = 9;
		Jitter = Jitter9;
		break;
	case 4:
	default:
		Jcount = 4;
		Jitter = Jitter4;
		break;
	}
}


extern bool useDetails = false;
float rayTrace	( Fvector& P, Fvector& D, float R, Face* skip)
{ 	
	if (useDetails)
		return EmbreeMain.RaytraceEmbreeDetails(P, D, R);
 	else 
		return EmbreeMain.RaytraceEmbreeProcess(P, D, R, skip);	
}

IC void LightPoint(  base_color_c &C, Fvector &P, Fvector &N, base_lighting& lights, u32 flags, Face* skip)
{ 
	auto ProcessLight = [&](R_Light& L, bool SunOrHemi) -> float
	{
		Fvector Ldir;
		Fvector Pnew = P;
		Pnew.mad(N, 0.01f);
		float add = 0.0f;
		switch (L.type)
		{
			case LT_DIRECT:
			{
				Ldir.invert(L.direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				float trace = rayTrace( Pnew, Ldir, 1000.f, skip);
				add = SunOrHemi ? L.energy * trace : D * L.energy * trace;
				break;
			}

			case LT_POINT:
			{
				float sqD = P.distance_to_sqr(L.position);
				if (sqD > L.range2)			break;

				Ldir.sub(L.position, P).normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				float R = _sqrt(sqD);
				float trace = rayTrace( Pnew, Ldir, R, skip);
				float scale = D * L.energy * trace;

				if (SunOrHemi)
				{
					add = scale / (L.attenuation0 + L.attenuation1 * R + L.attenuation2 * sqD);
				}
				else
				{
					add = scale * (1 / (L.attenuation0 + L.attenuation1 * R + L.attenuation2 * sqD) - R * L.falloff);
				}
				break;
			}
			
			case LT_SECONDARY:
			{
				float sqD = P.distance_to_sqr(L.position);
				if (sqD > L.range2)			break;

				Ldir.sub(L.position, P).normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				D *= -Ldir.dotproduct(L.direction);
				if (D <= 0)					break;

				float R = _sqrt(sqD);
				float trace = rayTrace( Pnew, Ldir, R, skip);
				add = powf(D, 0.125f) * L.energy * trace * (1 - R / L.range);
				break;
			}
		}

		return add;
	};

	// RGB Lights
	if (!(flags & LP_dont_rgb))
	{
		for (R_Light& L : lights.rgb)
		{
 			float accum = ProcessLight(L, false);
			C.rgb.x += accum * L.diffuse.x;
			C.rgb.y += accum * L.diffuse.y;
			C.rgb.z += accum * L.diffuse.z;
 		}
	}

	// Sun Lights
	if (!(flags & LP_dont_sun))
	{
		for (R_Light& L : lights.sun)
 			C.sun += ProcessLight(L, true);
 	}

	// Hemi Lights
	if (!(flags & LP_dont_hemi))
	{
		for (R_Light& L : lights.hemi)
 			C.hemi += ProcessLight(L, true);
 	}
} 


IC void LightPoint_Embree(EmbreeRayTraceModel* MDL, base_color_c& C, Fvector& P, Fvector& N, base_lighting& lights, u32 flags, Face* skip)
{
	auto ProcessLight = [&](R_Light& L, bool SunOrHemi) -> float
		{
			Fvector Ldir;
			Fvector Pnew = P;
			Pnew.mad(N, 0.01f);
			float add = 0.0f;
			switch (L.type)
			{
			case LT_DIRECT:
			{
				Ldir.invert(L.direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				float trace = MDL->RaytraceEmbreeProcess  ( Pnew, Ldir, 1000.f, skip);
				add = SunOrHemi ? L.energy * trace : D * L.energy * trace;
				break;
			}

			case LT_POINT:
			{
				float sqD = P.distance_to_sqr(L.position);
				if (sqD > L.range2)			break;

				Ldir.sub(L.position, P).normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				float R = _sqrt(sqD);
				float trace = MDL->RaytraceEmbreeProcess(Pnew, Ldir, R, skip);
				float scale = D * L.energy * trace;

				if (SunOrHemi)
				{
					add = scale / (L.attenuation0 + L.attenuation1 * R + L.attenuation2 * sqD);
				}
				else
				{
					add = scale * (1 / (L.attenuation0 + L.attenuation1 * R + L.attenuation2 * sqD) - R * L.falloff);
				}
				break;
			}

			case LT_SECONDARY:
			{
				float sqD = P.distance_to_sqr(L.position);
				if (sqD > L.range2)			break;

				Ldir.sub(L.position, P).normalize_safe();
				float D = Ldir.dotproduct(N);
				if (D <= 0)					break;

				D *= -Ldir.dotproduct(L.direction);
				if (D <= 0)					break;

				float R = _sqrt(sqD);
				float trace = MDL->RaytraceEmbreeProcess( Pnew, Ldir, R, skip);
				add = powf(D, 0.125f) * L.energy * trace * (1 - R / L.range);
				break;
			}
			}

			return add;
		};

	// RGB Lights
	if (!(flags & LP_dont_rgb))
	{
		for (R_Light& L : lights.rgb)
		{
			float accum = ProcessLight(L, false);
			C.rgb.x += accum * L.diffuse.x;
			C.rgb.y += accum * L.diffuse.y;
			C.rgb.z += accum * L.diffuse.z;
		}
	}

	// Sun Lights
	if (!(flags & LP_dont_sun))
	{
		for (R_Light& L : lights.sun)
			C.sun += ProcessLight(L, true);
	}

	// Hemi Lights
	if (!(flags & LP_dont_hemi))
	{
		for (R_Light& L : lights.hemi)
			C.hemi += ProcessLight(L, true);
	}
}