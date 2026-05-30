// Build.h: interface for the CBuild class.
//
//////////////////////////////////////////////////////////////////////
#pragma once

#include "../../xrcore/fs.h"
#include "../Public/shader_xrlc.h"

#include "../xrLCLight/b_build_texture.h"
#include "../xrLCLight/xrFace.h"
#include "../../xrcdb/xrcdb.h"

struct		 STextureParams;
extern "C" bool __declspec(dllimport)  DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);


class xrLC_GlobalData;
class xrMU_Model;
class xrMU_Reference;
class CDeflector;

extern "C" XRLC_LIGHT_API xrLC_GlobalData*	lc_global_data();
//////////////////////////////////////////////////////////////////////////
// tesselator callbacks



typedef	int		tesscb_estimator	( const Face* F );	// -1 = none, 0,1,2 = edge-number
typedef void	tesscb_face			(Face*		F);	// new face
typedef void	tesscb_vertex		(Vertex*	V);	// new vertex

class  base_lighting;
class  INetReader;

#include <psapi.h>

size_t GetHeapMemory();

//////////////////////////////////////////////////////////////////////////
class CBuild  
{
public:
	CMemoryWriter					&err_invalid();
	CMemoryWriter					&err_tjunction();
	CMemoryWriter					&err_multiedge();
	void							err_save	();

	Fbox							scene_bb;
	xr_vector<b_shader>				shader_render;
	xr_vector<b_shader>				shader_compile;
    xr_vector<b_light_dynamic>		L_dynamic;
	xr_vector<b_glow>				glows;
	xr_vector<b_portal>				portals;
	xr_vector<b_lod>				lods;
	string_path						path;
	xr_vector<LPCSTR>				g_Shaders;


	xr_vector<b_material>			&materials()	;
	xr_vector<b_BuildTexture>		&textures()	;
	base_lighting					&L_static();
	xr_vector<xrMU_Model*>			&mu_models();
	xr_vector<xrMU_Reference*>		&mu_refs();

	Shader_xrLC_LIB					&shaders();
 
	void	mem_Compact				();
public:
	void	Load					(const b_params& P, const IReader&  fs);
	void	Run						(LPCSTR path);
 
	void	RunAfterLight			( IWriter* fs	);
	void	PreOptimize				();
	void	CorrectTJunctions		();

	void	xrPhase_AdaptiveHT_tesselate();
	void	xrPhase_AdaptiveHT_calculate();

	void	u_Tesselate				(tesscb_estimator* E, tesscb_face* F, tesscb_vertex* V);
	void	u_SmoothVertColors		(int count);

	void	CalcNormals				();
	void	MU_ModelsCalculateNormals();
	void	xrPhase_TangentBasis	();

	void	BuildCForm				();
	void	BuildPortals			(IWriter &fs);
  		
	void	IsolateVertices			(BOOL bProgress);
	void	xrPhase_ResolveMaterials();
	void	xrPhase_UVmap			();
	void	xrPhase_Subdivide		();

	void	Light_prepare			();	
	void	Light					();

	void	LMaps					();
	void	ProcessLMAPS_CPU();
	
	void	xrPhase_SortDeflectors();
	void	xrPhase_MergeLM(xr_vector<CDeflector*>& deflectors);
 
	void	xrPhase_MergeGeometry	();

	void	Flex2OGF				();
	void	BuildSectors			();

	void	SaveLights				(IWriter &fs);
	void	SaveTREE				(IWriter &fs);
	void	SaveSectors				(IWriter &fs);

	void	validate_splits			();
	bool	IsOGFContainersEmpty	();
	void	CheckBeforeSave			( u32 stage );
	void	TempSave				( u32 stage );
 

	CBuild	();
	~CBuild	();
};

extern CBuild*							pBuild;			 
extern xr_vector<xr_vector<Face*>*>		g_XSplit;
