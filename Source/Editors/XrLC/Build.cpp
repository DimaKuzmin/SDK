// Build.cpp: implementation of the CBuild class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "build.h"
#include "../xrLCLight/xrMU_Model.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"
#include "../xrLCLight/mu_model_light.h"
#include "../XrLCLight/xrDeflector.h"
#include "../XrLCLight/xrMU_Model_Reference.h"
#include "../XrLCLight/cuda/xrDeflectorLight_Packed.h"
#include "OGF_Face.h"

void	export_ogf(xrMU_Reference& mu_mode);
void	calc_ogf(xrMU_Model& mu_model);
void	export_geometry	( xrMU_Model &	mu_model );
 
xr_vector<OGF_Base *>						g_tree;
xr_vector<xr_vector<Face*>*>				g_XSplit;
  
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

size_t GetHeapMemory()
{
	PROCESS_MEMORY_COUNTERS_EX pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
	{
		return pmc.PrivateUsage;
	}

	return 0;
}


//////////////////////////////////////////////////////////////////////

CBuild::CBuild()
{
}

CBuild::~CBuild()
{
	clMsg("mem usage start clearing:	%u mb", (u32(GetHeapMemory()) / 1024 / 1024));
	destroy_global_data();
 
 	for (auto OGF : g_tree)
 		xr_delete(OGF);
 	g_tree.clear();
	g_tree.shrink_to_fit();
	clMsg("mem usage g_tree clearing:	%u mb", (u32(GetHeapMemory()) / 1024 / 1024));

	for (auto faces : g_XSplit)
		xr_delete(faces);
	g_XSplit.clear();
	g_XSplit.shrink_to_fit();
	clMsg("mem usage g_XSplit clearing:	%u mb", (u32(GetHeapMemory()) / 1024 / 1024));
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
	for (auto F : lc_global_data()->g_faces())
		F->CacheOpacity();

	for (u32 m=0; m<mu_models().size(); m++)	
		mu_models()[m]->calc_faceopacity();
}
 
extern string_path LEVEL_PATH = "";
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
 
IC bool				FaceEqual(Face& F1, Face& F2);
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
  	PreOptimize();
	CorrectTJunctions();
  	xrPhase_AdaptiveHT_tesselate();

	Phase("Building collision database...");
	
	if (gCompilerMode.LC_Cforms)
	{
		BuildCForm();
		EmbreeMain.BuildRcast();
	}

	// Просщитывем освещение 
 	Light						();
 	RunAfterLight				( fs );
}
 
void CBuild::	RunAfterLight			( IWriter* fs	)
{
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
