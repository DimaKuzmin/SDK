#include "stdafx.h"

#include "xrlight_implicit.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrlight_implicitrun.h"

#include "tga.h"

#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"
 
#include "../../xrcdb/xrcdb.h"

#include "../../LauncherSDL/CompilersUI.h"

extern "C" bool __declspec(dllimport)  DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);

ImplicitCalcGlobs cl_globs;
DEF_MAP(Implicit,u32,ImplicitDeflector);

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
 		
		//se7kills PPL Style MT
		if (gCompilerMode.CUDA)
		{
			extern void RunImplicitCuda();
			RunImplicitCuda();
		}
		else
		{
			RunImplicitMultithread(defl);
		}

		defl.faces.clear();

		// Expand
		Status	("Processing lightmap...");
		for (u32 ref = 254; ref > 0; ref--)
		{
			if (!ApplyBorders(defl.lmap, ref))
				break;
		}

		Status	("Mixing lighting with texture...");
		{
			b_BuildTexture& TEX		=	*defl.texture;
			if (TEX.pSurface.Empty())
				Msg("[ImplicitLighting] Problem Texture : %s, Detected", * TEX.name);
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
			
			//se7kills rewrite		 
			sprintf(name, "%s",		lc_global_data()->level_path.c_str());

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
			string_path				name, out_name;
 			sprintf(name, "%s",		lc_global_data()->level_path.c_str());
			
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
