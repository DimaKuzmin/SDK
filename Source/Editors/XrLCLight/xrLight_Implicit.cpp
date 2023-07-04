#include "stdafx.h"

#include "xrlight_implicit.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrlight_implicitrun.h"

#include "tga.h"


#include "xrHardwareLight.h"


#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"
#include "xrlight_implicitcalcglobs.h"
#include "net_task_callback.h"

#include "../../xrcdb/xrcdb.h"



extern "C" bool __declspec(dllimport)  DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);

xrCriticalSection crImplicit;
int curHeight = 0;
ImplicitCalcGlobs cl_globs;

DEF_MAP(Implicit,u32,ImplicitDeflector);
 
void		ImplicitExecute::read			( INetReader	&r )
{
	y_start = r.r_u32();	
	y_end	= r.r_u32();
}
void		ImplicitExecute::write			( IWriter	&w ) const 
{
	R_ASSERT( y_start != (u32(-1)) );
	R_ASSERT( y_end != (u32(-1)) );
	w.w_u32( y_start );
	w.w_u32( y_end );
}



#include <algorithm>

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
		HardwareCalculator.PerformRaycast(RayRequests, (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | LP_UseFaceDisable, FinalColors, true);
		//finalize rays

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
						//Msg("X:%d, Y: %d == U: %d, V: %d ", LRequest.X, LRequest.Y, U, V);

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
 
void	ImplicitExecute::	receive_result			( INetReader	&r )
{
	R_ASSERT( y_start != (u32(-1)) );
	R_ASSERT( y_end != (u32(-1)) );
	ImplicitDeflector&		defl	= cl_globs.DATA();
	for (u32 V=y_start; V<y_end; V++)
	for (u32 U=0; U<defl.Width(); U++)
	{
				
		r_pod<base_color>( r, defl.Lumel( U, V ) );
		r_pod<u8>		 ( r, defl.Marker( U, V ) );

	}
}
void	ImplicitExecute::	send_result				( IWriter	&w ) const 
{
	R_ASSERT( y_start != (u32(-1)) );
	R_ASSERT( y_end != (u32(-1)) );
	ImplicitDeflector&		defl	= cl_globs.DATA();
	for (u32 V=y_start; V<y_end; V++)
	for (u32 U=0; U<defl.Width(); U++)
	{
		w_pod<base_color>( w, defl.Lumel( U, V ) );
		w_pod<u8>		 ( w, defl.Marker( U, V ) );
	}
}

void ImplicitExecute::clear()
{
	curHeight = 0;
}

void	ImplicitExecute::Execute(net_task_callback* net_callback)
{
	InitDB(&DB);
	

	net_cb = net_callback;
	//R_ASSERT( y_start != (u32(-1)) );
	//R_ASSERT( y_end != (u32(-1)) );
	ImplicitDeflector& defl = cl_globs.DATA();

	// Setup variables
	//u32		Jcount;
	//Fvector2	dim,half;
	//Fvector2	JS;
	//Fvector2*	Jitter;

	dim.set(float(defl.Width()), float(defl.Height()));
	half.set(.5f / dim.x, .5f / dim.y);

	// Jitter data
	JS.set(.499f / dim.x, .499f / dim.y);
	CTimer t;
	t.Start();
	Jitter_Select(Jitter, Jcount);

	// Lighting itself
	CDB::COLLIDER			DB;
	DB.ray_options(0);

	//for (u32 V=y_start; V<y_end; V++)

	for (;;)
	{
		int ID = 0;
		crImplicit.Enter();
		ID = curHeight;
		if (curHeight >= defl.Height())
		{
			crImplicit.Leave();
			break;
		}
		curHeight++;
		crImplicit.Leave();

		ForCycle(&defl, ID, TH_ID);
		if (ID % 128 == 0)
			Msg("CurV: %d, Sec[%.0f]", ID, t.GetElapsed_sec());
	}
}

extern u64 results;
extern u64 results_tDB;

float hemi;
float sun;

u32 cnt_good_hemi = 0;
u32 cnt_good_sun = 0;
u32 cnt_good_rgb = 0;

u32 cnt_bad_hemi = 0;
u32 cnt_bad_sun = 0;
u32 cnt_bad_rgb = 0;

bool SameFloat(float& a, float& b, float eps)
{
	if (a < (b - eps) || a >(b + eps))
		return false;
	else
		return true;
}

bool SameVector(Fvector& a, Fvector& b, float eps)
{
	if (a.x < (b.x - eps) || a.x >(b.x + eps))
		return false;

	if (a.y < (b.y - eps) || a.y >(b.y + eps))
		return false;
	
	if (a.z < (b.z - eps) || a.z >(b.z + eps))
		return false;

	return true;
}

//#define DEBUG_COLOR

void ImplicitExecute::ForCycle(ImplicitDeflector* defl, u32 V, int TH)
{
	for (u32 U = 0; U < defl->Width(); U++)
	{
		base_color_c	C;
#ifdef DEBUG_COLOR
		base_color_c	C_test;
#endif 
		u32				Fcount = 0;

		try
		{
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

						if (use_intel)
						{
							if (!inlc_global_data()->b_nosun())
								RaysToSUNLight_Deflector(TH, wP, wN, C, inlc_global_data()->L_static(), F);
							if (!inlc_global_data()->b_nohemi())
								RaysToHemiLight_Deflector(TH, wP, wN, C, inlc_global_data()->L_static(), F);
							if (!inlc_global_data()->b_norgb())
								RaysToRGBLight_Deflector(TH, wP, wN, C, inlc_global_data()->L_static(), F);
#ifdef DEBUG_COLOR						    
							LightPoint(&DB, inlc_global_data()->RCAST_Model(), C_test, wP, wN,
								inlc_global_data()->L_static(),
								(inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) |
								(inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) |
								(inlc_global_data()->b_nosun() ? LP_dont_sun : 0),
								F, false);

#endif
						    
							///Msg("RGB: %d, SUN: %d, HEMI: %d", lc_global_data()->b_norgb(), lc_global_data()->b_nosun(), lc_global_data()->b_nohemi());
						}
						else
						{
							LightPoint(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN,
								inlc_global_data()->L_static(),
								(inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) |
								(inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) |
								(inlc_global_data()->b_nosun() ? LP_dont_sun : 0),
								F, 1024);
						}
						 
						Fcount++;
					}
				}
			}
		}
		catch (...)
		{
			clMsg("* THREAD #%d: Access violation. Possibly recovered.");//,thID
		}
   
		if (Fcount)
		{

#ifdef DEBUG_COLOR
			if (!SameFloat(C.hemi, C_test.hemi, 0.2f))
			{
				Msg_IN_FILE("U: %d, V: %d, Hemi: %f != %f", U, V, C.hemi, C_test.hemi);
				cnt_bad_hemi++;
			}
			else
				cnt_good_hemi++;

			if (!SameFloat(C.sun, C_test.sun, 0.2f))
			{
				Msg_IN_FILE("U: %d, V: %d, sun: %f != %f", U, V, C.sun, C_test.sun);
				cnt_bad_sun++;
			}
			else
				cnt_good_sun++;

			if (!SameVector(C.rgb, C_test.rgb, 0.01f))
			{
				Msg_IN_FILE("U: %d, V: %d, RGB: [%f,%f,%f] != [%f,%f,%f]", U, V, VPUSH(C.rgb), VPUSH(C_test.rgb));
				cnt_bad_rgb++;
			}
			else
				cnt_good_rgb++;
#endif
			 
			//Msg("U: %d, V: %d, Hemi: %f, sun: %f, rgb: [%f][%f][%f]", U, V, C.hemi, C.sun, C.rgb.x, C.rgb.y, C.rgb.z);
			// Calculate lighting amount
	 
			C.scale(Fcount);
			C.mul(.5f);

			hemi += C.hemi;
			sun += C.sun;

			defl->Lumel(U, V)._set(C);
			defl->Marker(U, V) = 255;
		}
		else
		{
			defl->Marker(U, V) = 0;
		}

	}



	if (V % (64) == 0)
	{
#ifdef DEBUG_COLOR
		Msg("Total: HEMI: %f, SUN: %f", hemi, sun);
		Msg("SAME OPCODE: hemi: %d, sun: %d, rgb: %d", cnt_good_hemi, cnt_good_sun, cnt_good_rgb);
		Msg("DIFF OPCODE: hemi: %d, sun: %d, rgb: %d", cnt_bad_hemi, cnt_bad_sun, cnt_bad_rgb);
#endif
	}

}

void ImplicitLightingExec(BOOL b_net);
void ImplicitLightingTreadNetExec( void *p );
void ImplicitLighting(BOOL b_net)
{
	if (g_params().m_quality==ebqDraft) 
		return;
	if(!b_net)
	{
		ImplicitLightingExec(FALSE) ;
		return;
	}
	thread_spawn	(ImplicitLightingTreadNetExec,"worker-thread",1024*1024,0);

}
xrCriticalSection implicit_net_lock;
void XRLC_LIGHT_API ImplicitNetWait()
{
	implicit_net_lock.Enter();
	implicit_net_lock.Leave();
}
void ImplicitLightingTreadNetExec( void *p  )
{
	implicit_net_lock.Enter();
	ImplicitLightingExec(TRUE);
	implicit_net_lock.Leave();
}

static xr_vector<u32> not_clear;
void ImplicitLightingExec(BOOL b_net)
{
	
	Implicit		calculator;

	cl_globs.Allocate();
	not_clear.clear();

	// Sorting
	Status("Sorting faces...");
	for (vecFaceIt I=inlc_global_data()->g_faces().begin(); I!=inlc_global_data()->g_faces().end(); I++)
	{
		Face* F = *I;
		if (F->pDeflector)				continue;
		if (!F->hasImplicitLighting())	continue;
		
		Progress		(float(I-inlc_global_data()->g_faces().begin())/float(inlc_global_data()->g_faces().size()));
		b_material&		M	= inlc_global_data()->materials()[F->dwMaterial];
		u32				Tid = M.surfidx;
		b_BuildTexture*	T	= &(inlc_global_data()->textures()[Tid]);

		
 
		Implicit_it		it	= calculator.find(Tid);
		if (it==calculator.end()) 
		{
			ImplicitDeflector	ImpD;
			ImpD.texture		= T;
			ImpD.faces.push_back(F);
			calculator.insert	(mk_pair(Tid,ImpD));
			not_clear.push_back	(Tid);
		} 
		else 
		{
			ImplicitDeflector&	ImpD = it->second;
			ImpD.faces.push_back(F);
		}
	}

	
	// Lighing
	for (Implicit_it imp=calculator.begin(); imp!=calculator.end(); imp++)
	{
		ImplicitDeflector& defl = imp->second;
		Status			("Lighting implicit map '%s'...",defl.texture->name);
		Progress		(0);
		defl.Allocate	();
				
		// Setup cache
		Progress					(0);
		cl_globs.Initialize( defl );
		if(b_net)
			lc_net::RunImplicitnet( defl, not_clear );
		else
			RunImplicitMultithread(defl);
	
		defl.faces.clear_and_free();

		// Expand
		Status	("Processing lightmap...");
		for (u32 ref=254; ref>0; ref--)	if (!ApplyBorders(defl.lmap,ref)) break;

		Status	("Mixing lighting with texture...");
		{
			b_BuildTexture& TEX		=	*defl.texture;
			VERIFY					(!TEX.pSurface.Empty());
			u32*			color	= static_cast<u32*>(*TEX.pSurface);
			for (u32 V=0; V<defl.Height(); V++)	{
				for (u32 U=0; U<defl.Width(); U++)	{
					// Retreive Texel
					float	h	= defl.Lumel(U,V).h._r();
					u32 &C		= color[V*defl.Width() + U];
					C			= subst_alpha(C,u8_clr(h));
				}
			}
		}

		xr_vector<u32>				packed;
		defl.lmap.Pack				(packed);
		defl.Deallocate				();
		
		
		// base
		Status	("Saving base...");
		{
			string_path				name, out_name;
			sscanf					(strstr(Core.Params,"-f")+2,"%s",name);
			R_ASSERT				(name[0] && defl.texture);
			b_BuildTexture& TEX		=	*defl.texture;
			strconcat				(sizeof(out_name),out_name,name,"\\",TEX.name,".dds");
			FS.update_path			(out_name,"$game_levels$",out_name);
			clMsg					("Saving texture '%s'...",out_name);
			VerifyPath				(out_name);
			BYTE* raw_data			=	LPBYTE(*TEX.pSurface);
			u32	w					=	TEX.dwWidth;
			u32	h					=	TEX.dwHeight;
			u32	pitch				=	w*4;
			STextureParams			fmt	= TEX.THM;
			fmt.fmt					= STextureParams::tfDXT5;
			fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
			fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
			fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);
			DXTCompress				(out_name,raw_data,0,w,h,pitch,&fmt,4);
		}

		// lmap
		Status	("Saving lmap...");
		{
			//xr_vector<u32>			packed;
			//defl.lmap.Pack			(packed);

			string_path				name, out_name;
			sscanf					(strstr(GetCommandLine(),"-f")+2,"%s",name);
			b_BuildTexture& TEX		=	*defl.texture;
			strconcat				(sizeof(out_name),out_name,name,"\\",TEX.name,"_lm.dds");
			FS.update_path			(out_name,"$game_levels$",out_name);
			clMsg					("Saving texture '%s'...",out_name);
			VerifyPath				(out_name);
			BYTE* raw_data			= LPBYTE(&*packed.begin());
			u32	w					= TEX.dwWidth;
			u32	h					= TEX.dwHeight;
			u32	pitch				= w*4;
			STextureParams			fmt;
			fmt.fmt					= STextureParams::tfDXT5;
			fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
			fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
			fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);
			DXTCompress				(out_name,raw_data,0,w,h,pitch,&fmt,4);
		}
		//defl.Deallocate				();
	}
	not_clear.clear();
	cl_globs.Deallocate();
	calculator.clear	();
	if(b_net)
		inlc_global_data()->clear_build_textures_surface();
}
