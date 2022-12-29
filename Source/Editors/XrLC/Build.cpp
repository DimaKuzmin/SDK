// Build.cpp: implementation of the CBuild class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "build.h"

#include "../xrLCLight/xrMU_Model.h"



#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"
#include "../xrLCLight/mu_model_light.h"
#include "../xrLCLight/net_cl_data_prepare.h"
#include "../xrLCLight/serialize.h"
//#include "../xrLCLight/lcnet_task_manager.h"
void	calc_ogf		( xrMU_Model &	mu_model );
void	export_geometry	( xrMU_Model &	mu_model );

void	export_ogf		( xrMU_Reference& mu_reference );

using namespace			std;
struct OGF_Base;
xr_vector<OGF_Base *>	g_tree;

//BOOL					b_noise		= FALSE;
//BOOL					b_radiosity	= FALSE;
//BOOL					b_net_light	= FALSE;
SBuildOptions			g_build_options;
vec2Face				g_XSplit;
 
void	CBuild::CheckBeforeSave( u32 stage )
{
	bool b_g_tree_empty = g_tree.empty() ;
	R_ASSERT( b_g_tree_empty );
	bool b_g_XSplit_empty = g_XSplit.empty();
	R_ASSERT( b_g_XSplit_empty );
	bool b_IsOGFContainersEmpty = IsOGFContainersEmpty();
	R_ASSERT( b_IsOGFContainersEmpty );
	
	
	
}

void	CBuild::TempSave( u32 stage )
{
	CheckBeforeSave( stage );

}

	//Fbox								scene_bb;
	//xr_vector<b_shader>				shader_render;
	//xr_vector<b_shader>				shader_compile;
 //   xr_vector<b_light_dynamic>		L_dynamic;
	//xr_vector<b_glow>					glows;
	//xr_vector<b_portal>				portals;
	//xr_vector<b_lod>					lods;
	//string_path						path;
	//xr_vector<LPCSTR>					g_Shaders;
void	CBuild::read( INetReader &r )
{
	r_pod( r, g_build_options );

	r_pod( r, scene_bb );
	r_pod_vector( r, shader_render );
	r_pod_vector( r, shader_compile );
	r_pod_vector( r, L_dynamic );
	r_pod_vector( r, glows );
	r_pod_vector( r, portals );
	r_pod_vector( r, lods );
	r_pod( r, path );
	r_pod_vector( r, g_Shaders );
	
}

void	CBuild::write( IWriter	&w ) const 
{
	w_pod( w, g_build_options );
	
	w_pod( w, scene_bb );
	w_pod_vector( w, shader_render );
	w_pod_vector( w, shader_compile );
	w_pod_vector(  w,L_dynamic );
	w_pod_vector( w, glows );
	w_pod_vector( w, portals );
	w_pod_vector( w, lods );
	w_pod( w, path );
	w_pod_vector( w, g_Shaders );

}


//////////////////////////////////////////////////////////////////////

CBuild::CBuild()
{
	

}

CBuild::~CBuild()
{
	destroy_global_data();
}
 
CMemoryWriter&	CBuild::err_invalid()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_invalid(); 
}
CMemoryWriter&	CBuild::err_multiedge()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_multiedge(); 
}
CMemoryWriter	&CBuild::err_tjunction()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_tjunction(); 
}
xr_vector<b_material>&	CBuild::materials()	
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->materials(); 
}
xr_vector<b_BuildTexture>&	CBuild::textures()		
{
	VERIFY(lc_global_data());
	return lc_global_data()->textures(); 
}

base_lighting&	CBuild::L_static()
{
	VERIFY(lc_global_data()); return lc_global_data()->L_static(); 
}

Shader_xrLC_LIB&	CBuild::shaders()		
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->shaders(); 
}

extern u16		RegisterShader		(LPCSTR T);


void CBuild::Light_prepare()
{
	for (vecFaceIt I=lc_global_data()->g_faces().begin();	I!=lc_global_data()->g_faces().end(); I++) (*I)->CacheOpacity();
	for (u32 m=0; m<mu_models().size(); m++)	mu_models()[m]->calc_faceopacity();
}

#ifdef LOAD_GL_DATA
void net_light ();
#endif

bool skip_sectors = false;

extern string_path LEVEL_PATH;

void CBuild::Run(LPCSTR P)
{
	lc_global_data()->initialize();
#ifdef LOAD_GL_DATA
	net_light();
	return;
#endif

	bool CformOnly = false;

	if (strstr(Core.Params, "-cform"))
	{
		CformOnly = true;
	}

	if (strstr(Core.Params, "-no_sectors"))
		skip_sectors = true;
	//****************************************** Open Level
	strconcat(sizeof(path), path, P, "\\");

	xr_strcpy(LEVEL_PATH, path);

	string_path					lfn;
	IWriter* fs = FS.w_open(strconcat(sizeof(lfn), lfn, path, "level."));
	fs->open_chunk(fsL_HEADER);
	hdrLEVEL H;
	H.XRLC_version = XRCL_PRODUCTION_VERSION;
	H.XRLC_quality = g_params().m_quality;
	fs->w(&H, sizeof(H));
	fs->close_chunk();

	if (strstr(Core.Params, "-sample_9"))
		g_params().m_lm_jitter_samples = 9;
	else if (strstr(Core.Params, "-sample_4"))
		g_params().m_lm_jitter_samples = 4;
	else if (strstr(Core.Params, "-sample_1"))
		g_params().m_lm_jitter_samples = 1;

	LPCSTR pixel = strstr(Core.Params, "-pxpm");
	LPCSTR val = pixel + 5;
	int value;
	if (pixel && sscanf(val, "%d", &value))
	{
 		g_params().m_lm_pixels_per_meter = value;
	}

	bool use_avx = strstr(Core.Params, "-use_avx");
	bool use_sse = strstr(Core.Params, "-use_sse");
	bool use_fpu = strstr(Core.Params, "-use_fpu");

 	if (use_avx)
		g_params().ray_calc_type = 2;

	if (use_sse)
		g_params().ray_calc_type = 1;

	if (use_fpu)
		g_params().ray_calc_type = 0;

	 

	//****************************************** Dumb entry in shader-registration
	RegisterShader("");

	//****************************************** Saving lights
	{
		string256			fn;
		IWriter* fs = FS.w_open(strconcat(sizeof(fn), fn, pBuild->path, "build.lights"));
		fs->w_chunk(0, &*L_static().rgb.begin(), L_static().rgb.size() * sizeof(R_Light));
		fs->w_chunk(1, &*L_static().hemi.begin(), L_static().hemi.size() * sizeof(R_Light));
		fs->w_chunk(2, &*L_static().sun.begin(), L_static().sun.size() * sizeof(R_Light));
		FS.w_close(fs);
	}

	//****************************************** Optimizing + checking for T-junctions
	log_vminfo();
	
	FPU::m64r();
	Phase("Optimizing...");
	mem_Compact();
	if (!strstr(Core.Params, "-no_optimize"))
		PreOptimize();
	CorrectTJunctions();
	
	log_vminfo();

	if (strstr(Core.Params, "-no_adaptive") == 0 && !CformOnly)
	{
	//****************************************** HEMI-Tesselate
	
		FPU::m64r();
		Phase("Adaptive HT...");
		mem_Compact();
		log_vminfo();
		xrPhase_AdaptiveHT();
	}

	log_vminfo();

	//****************************************** Collision DB
	//should be after normals, so that double-sided faces gets separated
 
	//****************************************** Building normals
	FPU::m64r();
	Phase("Building normals...");
	mem_Compact();
	CalcNormals();
	//SmoothVertColors			(5);

	FPU::m64r					();
	Phase						("Building collision database...");
	mem_Compact					();
	log_vminfo();
	BuildCForm					();
	log_vminfo();
	if (CformOnly)
		return;

	BuildPortals(*fs);

	//****************************************** T-Basis
	{
		FPU::m64r();
		Phase("Building tangent-basis...");
		xrPhase_TangentBasis();
		mem_Compact();
	}
 
 
	//****************************************** GLOBAL-RayCast model
	FPU::m64r();
	Phase("Building rcast-CFORM model...");
	mem_Compact();
	log_vminfo();
	Light_prepare();
	BuildRapid(TRUE);
	log_vminfo();
 


	//****************************************** GLOBAL-ILLUMINATION
	if (g_build_options.b_radiosity)			
	{
		FPU::m64r					();
		Phase						("Radiosity-Solver...");
		mem_Compact					();
		Light_prepare				();
		xrPhase_Radiosity			();
	}

	//****************************************** Starting MU
	/* 	Moved TO After LIGHT (После стадии Convert To OGF и возможность задать ключом -mu_th потоки)	 (Возможно для сетевой компиляции стартуют раньше)
	FPU::m64r					();
	Phase						("LIGHT: Starting MU...");
	mem_Compact					();
	Light_prepare				();
	if(g_build_options.b_net_light)
	{
		lc_global_data()->mu_models_calc_materials();
		RunNetCompileDataPrepare( );
	}
	StartMu						();
	*/

	//****************************************** Resolve materials
	FPU::m64r					();
	Phase						("Resolving materials...");
	log_vminfo();
	mem_Compact					();
	xrPhase_ResolveMaterials	();
	IsolateVertices				(TRUE);

	log_vminfo();

	//****************************************** UV mapping
	{
		FPU::m64r					();
		Phase						("Build UV mapping...");
		
		mem_Compact					();
		log_vminfo();
		xrPhase_UVmap				();
		IsolateVertices				(TRUE);
	}
	
	log_vminfo();

	//****************************************** Subdivide geometry
	FPU::m64r					();
	Phase						("Subdividing geometry...");
	
	mem_Compact					();
	log_vminfo();
	xrPhase_Subdivide			();
	//IsolateVertices				(TRUE);
	lc_global_data()->vertices_isolate_and_pool_reload();

	log_vminfo();

	//****************************************** All lighting + lmaps building and saving
#ifdef NET_CMP
	mu_base.wait				(500);
	mu_secondary.wait			(500);
#endif

	if(g_build_options.b_net_light)
		SetGlobalLightmapsDataInitialized();
		
	Light						();
	RunAfterLight				( fs );

}
void	CBuild::StartMu	()
{
  //mu_base.start				(xr_new<CMUThread> (0));
  run_mu_light( !!g_build_options.b_net_light );
}
void CBuild::	RunAfterLight			( IWriter* fs	)
{
 	//****************************************** Merge geometry
	FPU::m64r					();
	Phase						("Merging geometry...");
	mem_Compact					();
	xrPhase_MergeGeometry		();

	//****************************************** Convert to OGF
	FPU::m64r					();
	Phase						("Converting to OGFs...");
	mem_Compact					();
	Flex2OGF					();

	//****************************************** Starting MU
	FPU::m64r();
	Phase("LIGHT: Starting MU...");
	mem_Compact();
	Light_prepare();
	if (g_build_options.b_net_light)
	{
		lc_global_data()->mu_models_calc_materials();
		RunNetCompileDataPrepare();
	}
	StartMu();
	 

	//****************************************** Wait for MU
	FPU::m64r					();
	Phase						("LIGHT: Waiting for MU-thread...");
	mem_Compact					();
	wait_mu_base				();

	if (!g_build_options.b_net_light)
	{
		Phase("LIGHT: Waiting for MU-Secondary threads...");
		wait_mu_secondary();
	}

	//****************************************** Export MU-models
	FPU::m64r					();
	Phase						("Converting MU-models to OGFs...");
	mem_Compact					();
	{
		u32 m;
		Status			("MU : Models...");
		for (m=0; m<mu_models().size(); m++)	
		{
			clMsg("ID[%d], size[%d]", m, mu_models().size());
			calc_ogf			(*mu_models()[m]);
			export_geometry		(*mu_models()[m]);
		}

		Status			("MU : References...");
		for (m = 0; m < mu_refs().size(); m++)
		{
			clMsg("muref ID[%d], size[%d]", m, mu_models().size());
			export_ogf(*mu_refs()[m]);
		}

//		lc_global_data()->clear_mu_models();
	}

	//****************************************** Destroy RCast-model
	FPU::m64r		();
	Phase			("Destroying ray-trace model...");
	mem_Compact		();
	lc_global_data()->destroy_rcmodel();
	
	if (!skip_sectors)
	{
		//****************************************** Build sectors
		FPU::m64r();
		Phase("Building sectors...");
		mem_Compact();
		BuildSectors();
	}

	//****************************************** Saving MISC stuff
	FPU::m64r		();
	Phase			("Saving...");
	mem_Compact		();
	SaveLights		(*fs);

	fs->open_chunk	(fsL_GLOWS);
	
	for (u32 i=0; i<glows.size(); i++)
	{
		b_glow&	G	= glows[i];
		fs->w		(&G,4*sizeof(float));
		string1024	sid;
		strconcat	(sizeof(sid),sid,
			shader_render[materials()[G.dwMaterial].shader].name,
			"/",
			textures()		[materials()[G.dwMaterial].surfidx].name
			);
		fs->w_u16	(RegisterShader(sid));
	}
	fs->close_chunk	();

	SaveTREE		(*fs);
	SaveSectors		(*fs);

	err_save		();
 
	mem_Compact();
}

void CBuild::err_save	()
{
	string_path		log_name;
	strconcat		(sizeof(log_name),log_name,"build_",Core.UserName,".err");
	FS.update_path	(log_name,"$logs$",log_name);

	IWriter*		fs	= FS.w_open(log_name);
	IWriter&		err = *fs;

	// t-junction
	err.open_chunk	(0);
	err.w_u32		(err_tjunction().size()/(1*sizeof(Fvector)));
	err.w			(err_tjunction().pointer(), err_tjunction().size());
	err.close_chunk	();

	// m-edje
	err.open_chunk	(1);
	err.w_u32		(err_multiedge().size()/(2*sizeof(Fvector)));
	err.w			(err_multiedge().pointer(), err_multiedge().size());
	err.close_chunk	();

	// invalid
	err.open_chunk	(2);
	err.w_u32		(err_invalid().size()/(3*sizeof(Fvector)));
	err.w			(err_invalid().pointer(), err_invalid().size());
	err.close_chunk	();

	FS.w_close( fs );
}

void CBuild::MU_ModelsCalculateNormals()
{
		for		(u32 m=0; m<mu_models().size(); m++)
			calc_normals( *mu_models()[m] );
}

xr_vector<xrMU_Model*>&CBuild::mu_models()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->mu_models(); 
}

xr_vector<xrMU_Reference*>&CBuild::mu_refs()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->mu_refs(); 
}

void CBuild::ImplicitLighting()
{
	::ImplicitLighting( g_build_options.b_net_light );
}