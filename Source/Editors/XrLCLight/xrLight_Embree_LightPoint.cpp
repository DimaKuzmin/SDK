#include "stdafx.h"
#include "xrDeflector.h"
#include "R_light.h"
#include "light_point.h"
#include "base_lighting.h"
#include "xrLC_GlobalData.h"

#include "xrLight_Embree.h"
  

float EmbreeRayTrace(R_Light& L, Fvector& P, Fvector& D, float R, Face* skip, BOOL bUseFaceDisable)
{
 	return RaytraceEmbreeProcess( L, P, D, R, skip);
}

void LightPointEmbree(  base_color_c& C, Fvector& P, Fvector& N, base_lighting& lights, u32 flags, void* data_skip)
{
	Fvector		Ldir, Pnew;
	Pnew.mad(P, N, 0.01f);

	BOOL		bUseFaceDisable = flags & LP_UseFaceDisable;
	float		MAX_DISTANCE = 1000.0f;

	Face* skip = (Face*) data_skip;

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

					// Trace Light
					float scale = D * L->energy * EmbreeRayTrace( *L, Pnew, Ldir, MAX_DISTANCE, skip, bUseFaceDisable);
					C.rgb.x += scale * L->diffuse.x;
					C.rgb.y += scale * L->diffuse.y;
					C.rgb.z += scale * L->diffuse.z;
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
					float R = _sqrt(sqD);
					float scale = D * L->energy * EmbreeRayTrace( *L, Pnew, Ldir, R, skip, bUseFaceDisable);
					float A;

					if (inlc_global_data()->gl_linear())
					{
						A = 1 - R / L->range;
					}
					else
					{
						//	Igor: let A equal 0 at the light boundary
						A = scale *
							(
								1 / (L->attenuation0 + L->attenuation1 * R + L->attenuation2 * sqD) -
								R * L->falloff
								);

					}

					C.rgb.x += A * L->diffuse.x;
					C.rgb.y += A * L->diffuse.y;
					C.rgb.z += A * L->diffuse.z;
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
					float scale = powf(D, 1.f / 8.f) * L->energy * EmbreeRayTrace( *L, Pnew, Ldir, R, skip, bUseFaceDisable);
					float A = scale * (1 - R / L->range);
					L->position = Psave;

					C.rgb.x += A * L->diffuse.x;
					C.rgb.y += A * L->diffuse.y;
					C.rgb.z += A * L->diffuse.z;
				}
				break;
			}
		}
	}

	if (0 == (flags & LP_dont_sun))
	{
 		R_Light* L = &*(lights.sun.begin()), * E = &*(lights.sun.end());
		for (; L != E; L++)
		{
			if (L->type == LT_DIRECT)
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(N);
				if (D <= 0) continue;

				// Trace Light
				float scale = L->energy * EmbreeRayTrace( *L, Pnew, Ldir, MAX_DISTANCE, skip, bUseFaceDisable);
				C.sun += scale;
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
				float R = _sqrt(sqD);
				float scale = D * L->energy * EmbreeRayTrace( *L, Pnew, Ldir, R, skip, bUseFaceDisable);
				float A = scale / (L->attenuation0 + L->attenuation1 * R + L->attenuation2 * sqD);

				C.sun += A;
			}
		}
	}

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


				// Trace Light
				Fvector		PMoved;
				PMoved.mad(Pnew, Ldir, 0.001f);
				float scale = L->energy * EmbreeRayTrace( *L, PMoved, Ldir, MAX_DISTANCE, skip, bUseFaceDisable);
				C.hemi += scale;
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

				// Trace Light
				float R = _sqrt(sqD);
				float scale = D * L->energy * EmbreeRayTrace( *L, Pnew, Ldir, R, skip, bUseFaceDisable);
				float A = scale / (L->attenuation0 + L->attenuation1 * R + L->attenuation2 * sqD);

				C.hemi += A;
			}

		}
	}
}