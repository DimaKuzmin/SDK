#include "stdafx.h"
//#include "build.h"
//#include "std_classes.h"
#include "xrThread.h"
#include "xrdeflector.h"
#include "xrlc_globaldata.h"
#include "light_point.h"
#include "xrface.h"
#include "net_task.h"

#include "xrHardwareLight.h"

extern void Jitter_Select	(Fvector2* &Jitter, u32& Jcount);

void CDeflector::L_Direct_Edge (CDB::COLLIDER* DB, base_lighting* LightsSelected, Fvector2& p1, Fvector2& p2, Fvector& v1, Fvector& v2, Fvector& N, float texel_size, Face* skip)
{
	Fvector		vdir;
	vdir.sub	(v2,v1);
	
	lm_layer&	lm	= layer;
	
	Fvector2		size; 
	size.x			= p2.x-p1.x;
	size.y			= p2.y-p1.y;
	int	du			= iCeil(_abs(size.x)/texel_size);
	int	dv			= iCeil(_abs(size.y)/texel_size);
	int steps		= _max(du,dv);
	if (steps<=0)	return;
	
	for (int I=0; I<=steps; I++)
	{
		float	time = float(I)/float(steps);
		Fvector2	uv;
		uv.x	= size.x*time+p1.x;
		uv.y	= size.y*time+p1.y;
		int	_x  = iFloor(uv.x*float(lm.width)); 
		int _y	= iFloor(uv.y*float(lm.height));
		
		if ((_x<0)||(_x>=(int)lm.width))	continue;
		if ((_y<0)||(_y>=(int)lm.height))	continue;
		
		if (lm.marker[_y*lm.width+_x])		continue;
		
		// ok - perform lighting
		base_color_c	C;
		Fvector			P;	P.mad(v1,vdir,time);
		VERIFY(inlc_global_data());
		VERIFY(inlc_global_data()->RCAST_Model());

		LightPoint		(DB, inlc_global_data()->RCAST_Model(), C, P, N, *LightsSelected, (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) |(inlc_global_data()->b_nosun()?LP_dont_sun:0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) |LP_DEFAULT, skip); //.
		
		C.mul		(.5f);
		lm.surface	[_y*lm.width+_x]._set	(C);
		lm.marker	[_y*lm.width+_x]		= 255;
	}
}

void CDeflector::L_Direct	(CDB::COLLIDER* DB, base_lighting* LightsSelected, HASH& H)
{
	R_ASSERT	(DB);
	R_ASSERT	(LightsSelected);

	lm_layer&	lm = layer;

	// Setup variables
	Fvector2	dim,half;
	dim.set		(float(lm.width),float(lm.height));
	half.set	(.5f/dim.x,.5f/dim.y);
	
	// Jitter data
	Fvector2	JS;
	JS.set		(.4999f/dim.x, .4999f/dim.y);
	
	u32			Jcount;
	Fvector2*	Jitter;
	Jitter_Select(Jitter, Jcount);
	
	// Lighting itself
	DB->ray_options	(0);
	
//	Msg("Height MAP[%d], W[%d]", lm.height, lm.width);
	
	for (u32 V=0; V<lm.height; V++)
	{
		if(_net_session && !_net_session->test_connection())
			 return;
		for (u32 U=0; U<lm.width; U++)	{
#ifdef NET_CMP
			if(V*lm.width+U!=8335)
				continue;
#endif
			u32				Fcount	= 0;
			base_color_c	C;
			


			try {
				for (u32 J=0; J<Jcount; J++) 
				{
					// LUMEL space
					Fvector2 P;
					P.x = float(U)/dim.x + half.x + Jitter[J].x * JS.x;
					P.y = float(V)/dim.y + half.y + Jitter[J].y * JS.y;
					
					xr_vector<UVtri*>&	space	= H.query(P.x,P.y);
					
					// World space
					Fvector		wP,wN,B;
					for (UVtri** it=&*space.begin(); it!=&*space.end(); it++)
					{
						if ((*it)->isInside(P,B) )
						{
							// We found triangle and have barycentric coords
							Face	*F	= (*it)->owner;
							Vertex	*V1 = F->v[0];
							Vertex	*V2 = F->v[1];
							Vertex	*V3 = F->v[2];
							wP.from_bary(V1->P,V2->P,V3->P,B);
//. �� ����� ������������	if (F->Shader().flags.bLIGHT_Sharp)	{ wN.set(F->N); }
//							else								
							{ 
								wN.from_bary(V1->N,V2->N,V3->N,B);	exact_normalize	(wN); 
								wN.add		(F->N);					exact_normalize	(wN);
							}
							
							if (!xrHardwareLight::IsEnabled())
							{
								try
								{
									VERIFY(inlc_global_data());
									VERIFY(inlc_global_data()->RCAST_Model());
									LightPoint(DB, inlc_global_data()->RCAST_Model(), C, wP, wN, *LightsSelected, (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | LP_UseFaceDisable, F); //.
									Fcount += 1;
								}
								catch (...) {
									clMsg("* ERROR (CDB). Recovered. ");
								}
							}
							else
							{
								lm.SurfaceLightRequests.push_back(LightpointRequest(U,V, wP, wN, F));
								lm.marker[V * lm.width + U] = 255;
							}
							

							break;
						}
					}
				} 
			} catch (...) {
				clMsg("* ERROR (Light). Recovered. ");
			}

			if (!xrHardwareLight::IsEnabled())
			{
				if (Fcount)
				{
					C.scale(Fcount);
					C.mul(.5f);
					lm.surface[V * lm.width + U]._set(C);
					lm.marker[V * lm.width + U] = 255;
				}
				else
				{
					lm.surface[V * lm.width + U]._set(C);	// 0-0-0-0-0
					lm.marker[V * lm.width + U] = 0;
				}
			}
		}
	}
	
	// *** Render Edges
	float texel_size = (1.f/float(_max(lm.width,lm.height)))/8.f;
	for (u32 t=0; t<UVpolys.size(); t++)
	{
		UVtri&		T	= UVpolys[t];
		Face*		F	= T.owner;
		R_ASSERT	(F);
		try {
			L_Direct_Edge	(DB,LightsSelected, T.uv[0], T.uv[1], F->v[0]->P, F->v[1]->P, F->N, texel_size,F);
			L_Direct_Edge	(DB,LightsSelected, T.uv[1], T.uv[2], F->v[1]->P, F->v[2]->P, F->N, texel_size,F);
			L_Direct_Edge	(DB,LightsSelected, T.uv[2], T.uv[0], F->v[2]->P, F->v[0]->P, F->N, texel_size,F);
		} catch (...)
		{
			clMsg("* ERROR (Edge). Recovered. ");
		}
	}


	GPU_Calculation();
}

void CDeflector::GPU_Calculation()
{
	if (xrHardwareLight::IsEnabled())
	{
		//cast and finalize
		if (layer.SurfaceLightRequests.size() == 0)
		{
			return;
		}
		xrHardwareLight& HardwareCalculator = xrHardwareLight::Get();

		//pack that shit in to task, but remember order
		xr_vector <RayRequest> RayRequests;
		size_t SurfaceCount = layer.SurfaceLightRequests.size();
		RayRequests.reserve(SurfaceCount);
		for (size_t SurfaceID = 0; SurfaceID < SurfaceCount; ++SurfaceID)
		{
			LightpointRequest& LRequest = layer.SurfaceLightRequests[SurfaceID];
			RayRequests.push_back(RayRequest{ LRequest.Position, LRequest.Normal, LRequest.FaceToSkip });
		}

		xr_vector<base_color_c> FinalColors;
		HardwareCalculator.PerformRaycast(RayRequests, (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | LP_UseFaceDisable, FinalColors);		
 
		//finalize rays

		//all that we must remember - we have fucking jitter. And that we don't have much time, because we have tons of that shit
		//#TODO: Invoke several threads!
		u32 SurfaceRequestCursor = 0;
		u32 AlmostMaxSurfaceLightRequest = (u32)layer.SurfaceLightRequests.size() - 1;
		for (u32 V = 0; V < layer.height; V++)
		{
			for (u32 U = 0; U < layer.width; U++)
			{
				LightpointRequest& LRequest = layer.SurfaceLightRequests[SurfaceRequestCursor];

				if (LRequest.X == U && LRequest.Y == V)
				{
					//accumulate all color and draw to the lmap
					base_color_c ReallyFinalColor;
					int ColorCount = 0;
					for (;;)
					{
						LRequest = layer.SurfaceLightRequests[SurfaceRequestCursor];

						if (LRequest.X != U || LRequest.Y != V || SurfaceRequestCursor == AlmostMaxSurfaceLightRequest)
						{
							ReallyFinalColor.scale(ColorCount);
							ReallyFinalColor.mul(0.5f);
							layer.surface[V * layer.width + U]._set(ReallyFinalColor);
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

		layer.SurfaceLightRequests.clear();
	}
}
