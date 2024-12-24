#include "stdafx.h"

#include "xrlight_implicit.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrlight_implicitrun.h"

#include "tga.h"

#ifndef DevCPU
#include "xrHardwareLight.h"
#endif


#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"
 
#include "../../xrcdb/xrcdb.h"
#include "BuildArgs.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;


extern "C" bool __declspec(dllimport)  DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);

xrCriticalSection crImplicit;

ImplicitCalcGlobs cl_globs;
 
DEF_MAP(Implicit,u32,ImplicitDeflector);

 
 
/** NETWORK PROCESS **/


u32 curHeight = 0;
void ImplicitExecute::clear()
{
	curHeight = 0;
}

/*** THREAD MAIN (ON CPU) ***/
 
 
xrCriticalSection csOpacity;
 
void ImplicitExecute::Execute()
{ 
	ImplicitDeflector& defl = cl_globs.DATA();
	 
	dim.set(float(defl.Width()), float(defl.Height()));
	half.set(.5f / dim.x, .5f / dim.y);

	// Jitter data
	JS.set(.499f / dim.x, .499f / dim.y);
	CTimer t;
	t.Start();
	Jitter_Select(Jitter, Jcount);

	// Lighting itself
 	DB.ray_options(0);
 	 
 	for (;;)
	{
		int V = 0;
			
		crImplicit.Enter();
		V = curHeight;
		if (curHeight >= defl.Height())
		{
			crImplicit.Leave();
			break;
		}
		curHeight++;
		crImplicit.Leave();

  
		// FOR CYCLE
 		for(u32 U = 0; U < defl.Width(); U++)
		{
			base_color_c	C;

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
							u32 flags = (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0);
							LightPoint(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN, inlc_global_data()->L_static(), flags, F);
							
							// LightPointEmbree( C,wP, wN, inlc_global_data()->L_static(), flags, F);

							Fcount++;
						}
					}
				}
			}
			catch (...)
			{
				clMsg("* THREAD #%d: Access violation. Possibly recovered."); 
			}

			if (Fcount)
			{
				// Calculate lighting amount
				C.scale(Fcount);
				C.mul(.5f);

				defl.Lumel(U, V)._set(C);
				defl.Marker(U, V) = 255;
			}
			else
			{
				defl.Marker(U, V) = 0;
			}

		}
		// FOR CYCLE END

		if (V % 128 == 0 || V == defl.Height())
		{
			clMsg("CurV: %d, Sec[%.0f]", V, t.GetElapsed_sec());
		}
	}
}
/** MAIN THREAD CALL EXECUTION, SORTING, SAVE**/

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
 		
		RunImplicitMultithread(defl);
						  
		defl.faces.clear_and_free();

		// Expand
		Status	("Processing lightmap...");
		for (u32 ref=254; ref>0; ref--)
		if (!ApplyBorders(defl.lmap,ref)) 
		break;

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
			
			//se7kills rewrite		//sscanf					(strstr(Core.Params,"-f")+2,"%s",name);
			sprintf(name, "%s", build_args->level_name.c_str());

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
			//sscanf					(strstr(GetCommandLine(),"-f")+2,"%s",name);
			sprintf(name, "%s", build_args->level_name.c_str());
			
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
 	
	cl_globs.Deallocate();
	calculator.clear	();	 
}

/** EXECUTION **/

void ImplicitLighting(BOOL b_net)
{
	if (g_params().m_quality == ebqDraft)
		return;

	ImplicitLightingExec(FALSE);
}
