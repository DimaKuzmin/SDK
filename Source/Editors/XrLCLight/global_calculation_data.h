#ifndef __GLOBAL_CALCULATION_DATA_H__
#define __GLOBAL_CALCULATION_DATA_H__

#include "../../editors/LevelEditor/Engine/communicate.h"
#include "base_lighting.h"
#include "global_slots_data.h"
#include "b_build_texture.h"
#include "global_slots_data.h"
#include "../../xrcdb/xrcdb.h"
class Shader_xrLC_LIB;

#include "embree_raytracing/EmbreeRayTrace.h"
//-----------------------------------------------------------------
struct global_claculation_data
{
	base_lighting					g_lights; 
	Shader_xrLC_LIB*				g_shaders_xrlc;
	b_params						g_params;
	xr_vector<b_material>			g_materials;
	xr_vector<b_BuildTexture>		g_textures;
	CDB::MODEL*						RCAST_Model;

	Fbox							LevelBB;
	global_slots_data				slots_data;
	xr_vector<b_shader>				g_shader_compile; 
   
			global_claculation_data		(): g_shaders_xrlc( 0 ) {}
	void	xrLoad						( );	
	void	xrUnload();

	xr_vector<FaceDataEmbree>			building_embree_faces;

	void	xrCalculateOpacity();
	void	xrLoadGeometry(IReader* fs);
};
extern global_claculation_data	gl_data;
//-----------------------------------------------------------------
#endif //__GLOBAL_CALCULATION_DATA_H__