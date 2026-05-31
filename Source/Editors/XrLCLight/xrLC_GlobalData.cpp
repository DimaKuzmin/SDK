#include "stdafx.h"

#include "xrLC_GlobalData.h"
#include "xrFace.h"
#include "xrdeflector.h"
#include "lightmap.h"
 
#include "mu_model_face.h"
#include "xrmu_model.h"
#include "xrmu_model_reference.h"
#include "../../xrcdb/xrcdb.h"

xrLC_GlobalData* data = 0;
  
xrLC_GlobalData*	lc_global_data()
{	
	return data;
}

void	create_global_data()
{
	VERIFY( !inlc_global_data() );
	data = xr_new<xrLC_GlobalData>();
}

void	destroy_global_data()
{
	VERIFY( inlc_global_data() );
	if(data)
		data->clear();
	xr_delete(data);
}

xrLC_GlobalData::xrLC_GlobalData	()  : b_vert_not_register( false )
{
}

void		xrLC_GlobalData	::				initialize		()
{
}

bool xrLC_GlobalData::b_r_vertices	()		
{
	return false;
}
 
xrLC_GlobalData::~xrLC_GlobalData()
{
}
 
template<typename T>
void vec_clear( xr_vector<T*> &v )
{
	typename xr_vector<T*>::iterator i = v.begin(), e = v.end();
	for(;i!=e;++i)
		xr_delete(*i);
	v.clear();
}

template<typename T>
void vec_spetial_clear( xr_vector<T> &v )
{
	typename xr_vector<T>::iterator i = v.begin(), e = v.end();
	for(;i!=e;++i)
		clear(*i);
	v.clear();
}

void mu_mesh_clear();
size_t GetHeapMemory();

void		xrLC_GlobalData::				clear			()
{
	vec_spetial_clear(_textures );
	_materials.clear();
	_shaders.Unload();
	clMsg("mem usage Textures clear:	%u mb",			(u32(GetHeapMemory()) / 1024 / 1024) );
 
	vec_clear(_g_lightmaps);
	clMsg("mem usage lmaps clear:		%u mb",			(u32(GetHeapMemory()) / 1024 / 1024));

 	vec_clear(_g_deflectors);
	clMsg("mem usage deflectors clear mesh: %u mb", (u32(GetHeapMemory()) / 1024 / 1024));

	// Geometry Buffers !
	vec_clear(_mu_models); 
	clMsg("mem usage _mu_models clear:	%u mb",			(u32(GetHeapMemory()) / 1024 / 1024));

	vec_clear(_mu_refs);
	clMsg("mem usage _mu_refs clear:	%u mb",			(u32(GetHeapMemory()) / 1024 / 1024));

	mu_mesh_clear();
	clMsg("mem usage mu clear mesh:		%u mb",			(u32(GetHeapMemory()) / 1024 / 1024));
	
 	gl_mesh_clear();
 	clMsg("mem usage static clear mesh: %u mb",			(u32(GetHeapMemory()) / 1024 / 1024));
}
