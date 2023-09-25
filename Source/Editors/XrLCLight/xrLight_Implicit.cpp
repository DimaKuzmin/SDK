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
#include <execution>
#include "xrRayDefinition.h"

/** NEW CUDA OPTIX FAST (NO RAM USAGE TO TEXTURES)  **/


float GetHitColor(u32 u, u32 v, u32 prim_id, Face* skip)
{
	CDB::MODEL* MDL = lc_global_data()->RCAST_Model();

  	// Access to texture
	CDB::TRI& clT										= MDL->get_tris()[prim_id];
	base_Face* F										= (base_Face*)(clT.pointer);
	if (0==F)										
		return 1;
	if (skip==F)										
		return 1;

	const Shader_xrLC&	SH								= F->Shader();
	if (!SH.flags.bLIGHT_CastShadow)					
		return 1;

	if (F->flags.bOpaque)	
	{
		// Opaque poly - cache it
		//L.tri[0].set	(rpinf.verts[0]);
		//L.tri[1].set	(rpinf.verts[1]);
		//L.tri[2].set	(rpinf.verts[2]);
		return 0;
	}

	b_material& M	= inlc_global_data()->materials()			[F->dwMaterial];
	b_texture&	T	= inlc_global_data()->textures()			[M.surfidx];

	if (T.pSurface.Empty())	
	{
		F->flags.bOpaque	= true;
		return 0;
	}

	// barycentric coords
	// note: W,U,V order
	Fvector B;
	B.set	(1.0f - u - v, u, v);

	// calc UV
	Fvector2*	cuv = F->getTC0					();
	Fvector2	uv;
	uv.x = cuv[0].x*B.x + cuv[1].x*B.y + cuv[2].x*B.z;
	uv.y = cuv[0].y*B.x + cuv[1].y*B.y + cuv[2].y*B.z;

	int U = iFloor(uv.x*float(T.dwWidth) + .5f);
	int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
	U %= T.dwWidth;		if (U<0) U+=T.dwWidth;
	V %= T.dwHeight;	if (V<0) V+=T.dwHeight;
	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel		= raw[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
	return opac;
 }
 
void GetLightColor(Ray& ray, RayAdditionalData& r_data, R_Light* L, float opacity, base_color_c& C)
{
	if (r_data.type_light == RLightTypes::RGB)
	{
		if (r_data.L->type == LT_DIRECT)
		{
			float scale = r_data.DotProduct * L->energy * opacity;
			C.rgb.x += scale * L->diffuse.x;
			C.rgb.y += scale * L->diffuse.y;
			C.rgb.z += scale * L->diffuse.z;			
		}

		if (r_data.L->type == LT_POINT)
		{
			float A;
			if (inlc_global_data()->gl_linear())
				A = 1 - ray.tmax / L->range;
			else
			{
 				float scale = r_data.DotProduct * L->energy * opacity;

				A = scale * 
				(	
					1 / (L->attenuation0 + L->attenuation1 * ray.tmax + L->attenuation2 * r_data.SquaredDir) - ray.tmax * L->falloff
				);
			}

			C.rgb.x += A * L->diffuse.x;
			C.rgb.y += A * L->diffuse.y;
			C.rgb.z += A * L->diffuse.z;		
		}

		if (r_data.L->type == LT_SECONDARY)
		{	
 			float scale = powf(r_data.DotProduct, 1.f / 8.f) * L->energy * opacity;
			float A = scale * (1 - ray.tmax / L->range);
			C.rgb.x += A * L->diffuse.x;
			C.rgb.y += A * L->diffuse.y;
			C.rgb.z += A * L->diffuse.z;		
		}
	}
 
	if (r_data.type_light == RLightTypes::HEMI)
	{
		if (r_data.L->type == LT_DIRECT)
		{				
 			float scale = L->energy * opacity;
			C.hemi += scale;			
		}

		if (r_data.L->type == LT_POINT)
		{  
			float scale = r_data.DotProduct * L->energy * opacity;
			float A			= scale / (L->attenuation0 + L->attenuation1 * ray.tmax + L->attenuation2 * r_data.SquaredDir);
			C.hemi += A;		
		}
 
	}

	if (r_data.type_light == RLightTypes::SUN)
	{
		if (r_data.L->type == LT_DIRECT)
		{
			float scale = L->energy * opacity;
			C.sun += scale;
		}

		if (r_data.L->type == LT_POINT)
		{
			float scale = r_data.DotProduct * L->energy * opacity;
			float A = scale / (L->attenuation0 + L->attenuation1 * ray.tmax + L->attenuation2 * r_data.SquaredDir);
			C.sun += A;		
		}
	}
}

extern void RaycastOptix(std::vector<Ray>& rays, std::vector<HitsResultVector>& outbuffer, size_t BUFFER_ID_MAX);
extern void GetLightsForOPTIX(LightpointRequest& reqvest, std::vector<Ray>& rays_buffer, std::vector<RayAdditionalData>& additional, int flag, size_t& BUFFER_ID) ;


void RunOptixThread(ImplicitDeflector& defl)
{
	//cast and finalize
	if (defl.lmap.SurfaceLightRequests.empty())
		return;

	//pack that shit in to task, but remember order
	
	CTimer t; 

	Msg("Start Getting Rays");

	int ids = 0;
 


	t.Start();

	u64 GlobalLightCPU = 0;
	u64 GlobalGPU_TIME = 0;

	CTimer tGlobal;
	tGlobal.Start();
	u32 prev_print = 0;

 
	std::vector<Ray> rays_buff;	
	rays_buff.resize(MAX_RAYS_TASK);
	std::vector<RayAdditionalData> rays_aditional;
	rays_aditional.resize(MAX_RAYS_TASK);
	
	u32 MAX_POSIBLE_RAYS =
		lc_global_data()->L_static().hemi.size() + lc_global_data()->L_static().sun.size() +lc_global_data()->L_static().sun.size();
	
	size_t IDS_BUFFER = 0;

	xr_vector<base_color_c> final_colors;
	final_colors.resize(defl.lmap.SizeArea());

	u32 IMPL_DEFL_Width = defl.lmap.width;

	for (LightpointRequest& reqvest : defl.lmap.SurfaceLightRequests)
	{
		ids++;
 
		if (IDS_BUFFER + MAX_POSIBLE_RAYS > MAX_RAYS_TASK)
		{
			GlobalLightCPU += t.GetElapsed_ms();

			StatusNoMSG("CPU_LIGHT: %d, GPU_TIME: %d, Update: %d/%d", 
				GlobalLightCPU,
				GlobalGPU_TIME,
				ids,
				defl.lmap.SurfaceLightRequests.size()
			);
		 
			Msg("Start RayTace: %u", IDS_BUFFER);
			t.Start();
			std::vector<HitsResultVector> hits;
			RaycastOptix(rays_buff, hits, IDS_BUFFER);
			GlobalGPU_TIME+=t.GetElapsed_ms();
 
			Msg("Start Collect Colors");

			// TODO MOVE TO LMAP COLORS
			for (u64 i = 0;  i < hits.size(); i++)
			{
				Ray& ray = rays_buff[i];
 				auto hits_reqvest = hits[i];
				RayAdditionalData& r_data = rays_aditional[i];
				int calc = r_data.V * IMPL_DEFL_Width + r_data.U;
				base_color_c& color = final_colors[calc];   

				//if (h256.count > 1)
				//	Msg_IN_FILE("Cnt: %d", h256.count);	

				for (int hit_id = 0; hit_id < hits_reqvest.hits.size(); hit_id++)
				{
					Hit& h = hits_reqvest.hits[hit_id];		
					
					//if (h.triId != -1 && h.triId != 0 && hit_id > 1)
					//	Msg_IN_FILE("BUFFERID[%d] HIT[%d]: {TRI: %d, u: %f, v: %f, Distance: %f} ", i, hit_id, h.triId, h.u, h.v, h.Distance);

					float opac = GetHitColor(h.u, h.v, h.triId, r_data.skip);
					if (opac == 0)
						break;
					GetLightColor(ray, r_data, r_data.L, opac, color);
				}
			}
			// TEST THIS

 			t.Start();
			IDS_BUFFER = 0;
		}

		GetLightsForOPTIX(reqvest, rays_buff, rays_aditional, 0, IDS_BUFFER);
	}

	for (int V = 0; V < defl.lmap.height; V++)
	{
		for (int U = 0; U < defl.lmap.width; U++)
		{
			int column = V * IMPL_DEFL_Width + U;
			defl.Lumel(U, V)._set(final_colors[column]);
		}
	}
}


/** CUDA DEFINATION **/

void FinalizeImplicit(ImplicitDeflector& defl, xr_vector<base_color_c>& FinalColors )
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

void RunCudaThread(bool optix)
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

	if (!optix)
		CalculateGPU(defl);
	else 
		RunOptixThread(defl);
}
 
 
/** NETWORK PROCESS **/

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

/*** THREAD MAIN (ON CPU) ***/

 
xrCriticalSection csOpacity;

void ImplicitExecute::Execute(net_task_callback* net_callback)
{
	net_cb = net_callback;
 
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
 	DB.ray_options(0);


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
		{	
			Msg("CurV: %d, Sec[%.0f]", ID, t.GetElapsed_sec());
		}
	}
}
  
void ImplicitExecute::ForCycle(ImplicitDeflector* defl, u32 V, int TH)
{
	for (u32 U = 0; U < defl->Width(); U++)
	{
		base_color_c	C;

		u32				Fcount = 0;

		//xr_map<int, LightpointRequestC> map;

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
 
						LightPoint(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN,
							inlc_global_data()->L_static(),
							(inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) |
							(inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) |
							(inlc_global_data()->b_nosun() ? LP_dont_sun : 0),
							F);
						 
						
						
						  
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
			// Calculate lighting amount
			C.scale(Fcount);
			C.mul(.5f);

			defl->Lumel(U, V)._set(C);
			defl->Marker(U, V) = 255;
		}
		else
		{
			defl->Marker(U, V) = 0;
		}
 
	}
}


/** EXECUTION **/


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
		if(b_net)
			lc_net::RunImplicitnet( defl, not_clear );
		else
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
