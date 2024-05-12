#include "stdafx.h"

#include "xrlight_implicit.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrlight_implicitrun.h"

#include "tga.h"
 
#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"
#include "xrlight_implicitcalcglobs.h"
 
#include "../../xrcdb/xrcdb.h"

//#include "xrlight_implicitrun.h"
#include "xrThread.h"
  

extern "C" bool __declspec(dllimport)  DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);



DEF_MAP(Implicit,u32,ImplicitDeflector);



#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

ImplicitCalcGlobs cl_globs;

int CurrentY = 0;
xrCriticalSection csImplicit;

class ImplicitThread : public CThread
{
public:

	ImplicitExecute		execute;
	ImplicitThread(u32 ID, u32 MAX_H ) : CThread(ID), execute(MAX_H)
	{

	}
	virtual void		Execute();


};

void	ImplicitThread::Execute()
{
	// Priority
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	Sleep(0);
	execute.Execute(thID, thProgress);
}
 
void RunImplicitMultithread(ImplicitDeflector& defl)
{
	// Start threads
	CThreadManager			tmanager;
	u32 MAX_THREADS = build_args->use_threads;
	CurrentY = 0;

	for (u32 thID = 0; thID < MAX_THREADS; thID++)
		tmanager.start(xr_new<ImplicitThread>( thID, defl.Height() ));
	tmanager.wait();


}
 
void	ImplicitExecute::	Execute	(int thID, volatile float& thProgress)
{
		ImplicitDeflector&		defl	= cl_globs.DATA();
		CDB::COLLIDER			DB;
		
		// Setup variables
		Fvector2	dim,half;
		dim.set		(float(defl.Width()),float(defl.Height()));
		half.set	(.5f/dim.x,.5f/dim.y);
		
		// Jitter data
		Fvector2	JS;
		JS.set		(.499f/dim.x, .499f/dim.y);
		u32			Jcount;
		Fvector2*	Jitter;
		Jitter_Select(Jitter, Jcount);
		
		// Lighting itself
		DB.ray_options	(0);



		//for (u32 V=y_start; V<y_end; V++)
		float last_progress = 0;

		while( true)
		{

			csImplicit.Enter();
			int V = CurrentY;
			
			if (V % 8 == 0)
 			{
 				float initial = 0;
				float progress = V / MAX_HEIGHT;
				//if (thID == 0 && progress > last_progress)
				{
					thProgress = progress;
					//Progress(initial + progress);
					last_progress = progress + 0.05f;
				}
 			}

			if (V % 512 == 0)
				Status("Implicit: %d \\ %d", V, MAX_HEIGHT);

			if (V >= MAX_HEIGHT)
			{
				csImplicit.Leave();
				break;
			}
			CurrentY ++;
  
			csImplicit.Leave();

			

			for (u32 U=0; U<defl.Width(); U++)
			{
 				base_color_c	C;
				u32				Fcount	= 0;
				
				try {
					for (u32 J=0; J<Jcount; J++) 
					{
						// LUMEL space
						Fvector2				P;
						P.x						= float(U)/dim.x + half.x + Jitter[J].x * JS.x;
						P.y						= float(V)/dim.y + half.y + Jitter[J].y * JS.y;
						xr_vector<Face*>& space	= cl_globs.Hash().query(P.x,P.y);
						
						// World space
						Fvector wP,wN,B;
						for (vecFaceIt it=space.begin(); it!=space.end(); it++)
						{
							Face	*F	= *it;
							_TCF&	tc	= F->tc[0];
							if (tc.isInside(P,B)) 
							{
								// We found triangle and have barycentric coords
								Vertex	*V1 = F->v[0];
								Vertex	*V2 = F->v[1];
								Vertex	*V3 = F->v[2];
								wP.from_bary(V1->P,V2->P,V3->P,B);
								wN.from_bary(V1->N,V2->N,V3->N,B);
								wN.normalize();
								LightPoint	(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN, inlc_global_data()->L_static(), (inlc_global_data()->b_nosun()?LP_dont_sun:0), F);
								Fcount		++;
							}
						}
					} 
				} catch (...)
				{
					clMsg("* THREAD #%d: Access violation. Possibly recovered.");//,thID
				}
				if (Fcount) {
					// Calculate lighting amount
					C.scale				(Fcount);
					C.mul				(.5f);
					defl.Lumel(U,V)._set(C);
					defl.Marker(U,V)	= 255;
				} else {
					defl.Marker(U,V)	= 0;
				}
			}
	//		thProgress	= float(V - y_start) / float(y_end-y_start);
		}
	}

 
void ImplicitLightingExec()
{
	
	Implicit		calculator;

	cl_globs.Allocate();
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
			 
		} else {
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
		for (u32 ref=254; ref>0; ref--)	if (!ApplyBorders(defl.lmap,ref)) break;

		Status	("Mixing lighting with texture...");
		{
			b_BuildTexture& TEX		=	*defl.texture;
			VERIFY					(!TEX.pSurface.Empty());
			u32* color = TEX.pSurface; //static_cast<u32*>(*TEX.pSurface);
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
			//sscanf					(strstr(Core.Params,"-f")+2,"%s",name);
			xr_strcpy(name, build_args->level_name.c_str());

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
			xr_strcpy(name, build_args->level_name.c_str());
			
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

void ImplicitLighting()
{
	if (g_params().m_quality == ebqDraft)
		return;

	ImplicitLightingExec();
}