#include "stdafx.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrDeflector.h"
#include "xrFace.h"
#include "xrLC_GlobalData.h"
#include "light_point.h"

/** CUDA DEFINATION **/
#ifndef DevCPU 
#include "xrHardwareLight.h"


void FinalizeImplicit(ImplicitDeflector& defl, xr_vector<base_color_c>& FinalColors)
{
	//all that we must remember - we have fucking jitter. And that we don't have much time, because we have tons of that shit
	u32 SurfaceRequestCursor = 0;
	u32 AlmostMaxSurfaceLightRequest = defl.lmap.SurfaceLightRequests.size() - 1;
	for (u32 V = 0; V < defl.lmap.height; V++)
	{
		for (u32 U = 0; U < defl.lmap.width; U++)
		{
			LightpointRequest& LRequest = defl.lmap.SurfaceLightRequests[SurfaceRequestCursor];

			if (LRequest.X == U && LRequest.Y == V)
			{
				//accumulate all color and draw to the lmap
				base_color_c ReallyFinalColor;
				int ColorCount = 0;
				for (;;)
				{
					LRequest = defl.lmap.SurfaceLightRequests[SurfaceRequestCursor];

					if (LRequest.X != U || LRequest.Y != V || SurfaceRequestCursor == AlmostMaxSurfaceLightRequest)
					{
						ReallyFinalColor.scale(ColorCount);
						ReallyFinalColor.mul(0.5f);
						defl.Lumel(U, V)._set(ReallyFinalColor);
						break;
					}

					base_color_c& CurrentColor = FinalColors[SurfaceRequestCursor];
					ReallyFinalColor.add(CurrentColor);

					++SurfaceRequestCursor;
					++ColorCount;
				}
			}
		}
	}

	defl.lmap.SurfaceLightRequests.clear();
}

void CalculateGPU(ImplicitDeflector& defl)
{
	Msg("CalculateGPU");
	if (true)
	{
		//cast and finalize
		if (defl.lmap.SurfaceLightRequests.empty())
		{
			return;
		}
		xrHardwareLight& HardwareCalculator = xrHardwareLight::Get();

		//pack that shit in to task, but remember order
		xr_vector <RayRequest> RayRequests;
		u32 SurfaceCount = defl.lmap.SurfaceLightRequests.size();
		RayRequests.reserve(SurfaceCount);
		for (int SurfaceID = 0; SurfaceID < SurfaceCount; ++SurfaceID)
		{
			LightpointRequest& LRequest = defl.lmap.SurfaceLightRequests[SurfaceID];
			RayRequests.push_back(RayRequest{ LRequest.Position, LRequest.Normal, LRequest.FaceToSkip });
		}

		xr_vector<base_color_c> FinalColors;
		HardwareCalculator.PerformRaycast(RayRequests, (inlc_global_data()->b_nosun() ? LP_dont_sun : 0), FinalColors, true);


		//finalize rays
		FinalizeImplicit(defl, FinalColors);
	}
}

void RunCudaThread()
{
	ImplicitDeflector& defl = cl_globs.DATA();
	CDB::COLLIDER			DB;

	// Setup variables
	Fvector2	dim, half;
	dim.set(float(defl.Width()), float(defl.Height()));
	half.set(.5f / dim.x, .5f / dim.y);

	// Jitter data
	Fvector2	JS;
	JS.set(.499f / dim.x, .499f / dim.y);
	u32			Jcount;
	Fvector2* Jitter;
	Jitter_Select(Jitter, Jcount);

	// Lighting itself
	DB.ray_options(0);
	for (u32 V = 0; V < defl.Height(); V++)
	{
		if (V % 128 == 0)
			Msg("CurV: %d", V);
		for (u32 U = 0; U < defl.Width(); U++)
		{
			base_color_c	C;
			u32				Fcount = 0;

			for (u32 J = 0; J < Jcount; J++)
			{
				// LUMEL space
				Fvector2				P;
				P.x = float(U) / dim.x + half.x + Jitter[J].x * JS.x;
				P.y = float(V) / dim.y + half.y + Jitter[J].y * JS.y;
				xr_vector<Face*>& space = cl_globs.Hash().query(P.x, P.y);

				// World space
				Fvector wP, wN, B;
				for (vecFaceIt it = space.begin(); it != space.end(); it++)
				{
					Face* F = *it;
					_TCF& tc = F->tc[0];
					if (tc.isInside(P, B))
					{
						// We found triangle and have barycentric coords
						Vertex* V1 = F->v[0];
						Vertex* V2 = F->v[1];
						Vertex* V3 = F->v[2];
						wP.from_bary(V1->P, V2->P, V3->P, B);
						wN.from_bary(V1->N, V2->N, V3->N, B);
						wN.normalize();

						defl.lmap.SurfaceLightRequests.emplace_back(U, V, wP, wN, F);
						defl.Marker(U, V) = 255;
						Fcount++;
					}
				}
			}
		}
	}

	CalculateGPU(defl);
}

#endif