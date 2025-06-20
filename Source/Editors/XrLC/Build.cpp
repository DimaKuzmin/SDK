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
 
using namespace			std;
struct OGF_Base;
SBuildOptions			g_build_options;

xr_vector<OGF_Base *>	g_tree;
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
	clMsg( "Stage: %s * [win32]: free[%u MB], reserved[%u MB], committed[%u MB]",
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
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
 
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

	Phase("Optimizing...");
  	// PreOptimize();
	CorrectTJunctions();
	
 	Light						();
 	RunAfterLight				( fs );
}
 
void CBuild::	RunAfterLight			( IWriter* fs	)
{
 	//****************************************** T-Basis
 	Phase("Building tangent-basis...");
	xrPhase_TangentBasis();
 
	// Tangent Basis To Convert OGF
	BuildPortals(*fs);

	//****************************************** Convert to OGF
 	Phase("Converting to OGFs...");
 	Flex2OGF();

	//****************************************** Export MU-models
 	Phase						("Converting MU-models to OGFs...");
 	Status			("MU : Models...");
	for (u32 m=0; m<mu_models().size(); m++)	
	{
		calc_ogf			(*mu_models()[m]);
		export_geometry		(*mu_models()[m]);
	}

	Status			("MU : References...");
	for (u32 m = 0; m < mu_refs().size(); m++)
	{
		StatusNoMSG("References [%d]/[%d]", m, mu_models().size());
		export_ogf(*mu_refs()[m]);
	}
 
	//****************************************** Destroy RCast-model
 	Phase			("Destroying ray-trace model...");
 	lc_global_data()->destroy_rcmodel();
 
	//****************************************** Build sectors
 	Phase("Building sectors...");
 	BuildSectors();

	//****************************************** Saving MISC stuff
 	Phase			("Saving...");
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