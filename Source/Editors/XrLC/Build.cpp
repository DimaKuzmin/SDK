// Build.cpp: implementation of the CBuild class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "build.h"

#include "../xrLCLight/xrMU_Model.h"



#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"
#include "../xrLCLight/mu_model_light.h"
 
//#include "../xrLCLight/lcnet_task_manager.h"
void	calc_ogf		( xrMU_Model &	mu_model );
void	export_geometry	( xrMU_Model &	mu_model );

void	export_ogf		( xrMU_Reference& mu_reference );



#include "../XrLCLight/BuildArgs.h"

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

using namespace			std;
struct OGF_Base;
SBuildOptions			g_build_options;

xr_vector<OGF_Base *>	g_tree;
vec2Face				g_XSplit;


//BOOL					b_noise		= FALSE;
//BOOL					b_radiosity	= FALSE;
//BOOL					b_net_light	= FALSE;


 
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
	for (vecFaceIt I=lc_global_data()->g_faces().begin();	I!=lc_global_data()->g_faces().end(); I++) 
		(*I)->CacheOpacity();

	for (u32 m=0; m<mu_models().size(); m++)	
		mu_models()[m]->calc_faceopacity();
}

#ifdef LOAD_GL_DATA
void net_light ();
#endif

 
extern string_path LEVEL_PATH = "";

#include "..\XrLCLight\xrHardwareLight.h"



void log_vminfo_new(LPCSTR stage)
{
	size_t  w_free, w_reserved, w_committed;
	vminfo(&w_free, &w_reserved, &w_committed);
	clMsg(
		"Stage: %s * [win32]: free[%u MB], reserved[%u MB], committed[%u MB]",
		stage,
		w_free / 1024 / 1024,
		w_reserved / 1024 / 1024,
		w_committed / 1024 / 1024
	);
}
 
#include "../XrLCLight/xrDeflector.h"
 
IC bool				FaceEqual(Face& F1, Face& F2);
#include "../XrLCLight/xrMU_Model_Reference.h"
 
void CBuild::Run(LPCSTR P)
{
	lc_global_data()->initialize();
#ifdef LOAD_GL_DATA
	net_light();
	return;
#endif

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);

	bool CformOnly = false;
	 
#pragma todo("se7kills TODO CFORM BUILD PARAMS ")

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
	log_vminfo_new("Loading");
	
	FPU::m64r();
	Phase("Optimizing...");
	mem_Compact();
	if (!build_args->no_optimize)
		PreOptimize();
	CorrectTJunctions();
	
	log_vminfo_new("Optimize");

	log_vminfo_new("Adaptive HT memory_pre: ");
	
	if (!CformOnly )
	{
 	//****************************************** HEMI-Tesselate
		FPU::m64r();
		Phase("Adaptive HT...");
		mem_Compact();
 		xrPhase_AdaptiveHT();
	}

	log_vminfo_new("Adaptive HT memory_after: ");

	//****************************************** Collision DB
	//should be after normals, so that double-sided faces gets separated
 
	//****************************************** Building normals
	FPU::m64r();
	Phase("Building normals...");
	mem_Compact();
	CalcNormals();
	//SmoothVertColors			(5);
	log_vminfo_new("Normals Memory: ");

	FPU::m64r					();
	Phase						("Building collision database...");
	mem_Compact					();
 	BuildCForm					();
	log_vminfo_new("CFORM Data");
	if (CformOnly)
		return;
 
	//****************************************** GLOBAL-RayCast model
	FPU::m64r();
	Phase("Building rcast-CFORM model...");

	log_vminfo_new("rcast-CFORM model memory_pre: ");

	mem_Compact();
 	Light_prepare();
	
	if (build_args->use_embree)
		BuildIntelModel(TRUE);
	else 
		BuildRapid(TRUE);
	
	log_vminfo_new("rcast-CFORM model momory_after");
 


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
	/* 	Moved TO After LIGHT (После стадии Convert To OGF и возможность задать ключом -th потоки)	 (Возможно для сетевой компиляции стартуют раньше)
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
 	mem_Compact					();
	xrPhase_ResolveMaterials	();
	IsolateVertices				(TRUE);

	log_vminfo_new("Resolving materials");

	//****************************************** UV mapping
 	FPU::m64r					();
	Phase						("Build UV mapping...");
 	mem_Compact					();
 	xrPhase_UVmap				();
	IsolateVertices				(TRUE);
 	
	log_vminfo_new("Build UV mapping");

	//****************************************** Subdivide geometry
  	FPU::m64r					();
  	Phase						("Subdividing geometry...");
   	mem_Compact					();
   	xrPhase_Subdivide			();
    log_vminfo_new("Subdividing geometry");

	// Se7Kills
	// Export Model DEFLECTORS 
	// ExportDeflectors();


	// Se7Kills Opacity BUFFERS

	//****************************************** All lighting + lmaps building and saving
 		
	Light						();
	RunAfterLight				( fs );

}
 
void CBuild::	RunAfterLight			( IWriter* fs	)
{
 	//****************************************** Merge geometry
	FPU::m64r					();
	Phase						("Merging geometry...");
	mem_Compact					();
	xrPhase_MergeGeometry		();
 	log_vminfo_new("Merging geometry");
	 

	// Tangent Basis To Convert OGF
	BuildPortals(*fs);
 	//****************************************** T-Basis
	{
		FPU::m64r();
		Phase("Building tangent-basis...");
		xrPhase_TangentBasis();
		mem_Compact();
	}
	log_vminfo_new("Tangents Memory: ");

	//****************************************** Convert to OGF
	FPU::m64r();
	Phase("Converting to OGFs...");
	mem_Compact();
	Flex2OGF();
 	log_vminfo_new("Converting to OGFs");

	//****************************************** Export MU-models
	FPU::m64r					();
	Phase						("Converting MU-models to OGFs...");
	mem_Compact					();
	{
		u32 m;
		Status			("MU : Models...");
		for (m=0; m<mu_models().size(); m++)	
		{
		//	clMsg("ID[%d], size[%d]", m, mu_models().size());
			calc_ogf			(*mu_models()[m]);
			export_geometry		(*mu_models()[m]);
		}

		Status			("MU : References...");
		for (m = 0; m < mu_refs().size(); m++)
		{
		//	clMsg("muref ID[%d], size[%d]", m, mu_models().size());
			export_ogf(*mu_refs()[m]);
		}
	}

	log_vminfo_new("Converting to mu-OGFs");

	//****************************************** Destroy RCast-model
	FPU::m64r		();
	Phase			("Destroying ray-trace model...");
	mem_Compact		();
	lc_global_data()->destroy_rcmodel();
	log_vminfo_new("Destroying ray-trace model");
 
	//****************************************** Build sectors
	FPU::m64r();
	Phase("Building sectors...");
	mem_Compact();
	BuildSectors();
	log_vminfo_new("Building sectors");

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