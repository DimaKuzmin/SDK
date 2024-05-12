#include "stdafx.h"

#include "xrLC_GlobalData.h"
#include "xrface.h"
#include "xrdeflector.h"
#include "lightmap.h"
#include "serialize.h"
#include "mu_model_face.h"
#include "xrmu_model.h"
#include "xrmu_model_reference.h"
#include "../../xrcdb/xrcdb.h"

bool g_using_smooth_groups = true;
bool g_smooth_groups_by_faces = false;

xrLC_GlobalData* data =0;

 

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


xrLC_GlobalData::xrLC_GlobalData	():
 _b_nosun(false),_gl_linear(false),
	b_vert_not_register( false )
{
	
	_cl_globs._RCAST_Model = 0;
 
}

void	xrLC_GlobalData	::destroy_rcmodel	()
{
	xr_delete		(_cl_globs._RCAST_Model);
}
void xrLC_GlobalData::clear_build_textures_surface()
{
	clLog( "mem usage before clear build textures surface: %u", Memory.mem_usage() );
	//xr_vector<b_BuildTexture>		_textures;
	xr_vector<b_BuildTexture>::iterator i = textures().begin();
	xr_vector<b_BuildTexture>::const_iterator e = textures().end();
	for(;i!=e;++i)
		::clear((*i));
	Memory.mem_compact();
	clLog( "mem usage after clear build textures surface: %u", Memory.mem_usage() );
}
void xrLC_GlobalData::clear_build_textures_surface( const xr_vector<u32> &exept )
{
	clLog( "mem usage before clear build textures surface: %u", Memory.mem_usage() );
	xr_vector<b_BuildTexture>::iterator i = textures().begin();
	xr_vector<b_BuildTexture>::const_iterator e = textures().end();
	xr_vector<b_BuildTexture>::const_iterator b = textures().begin();
	for(;i!=e;++i)
	{
		xr_vector<u32>::const_iterator ff = std::find( exept.begin(), exept.end(),u32( i - b ) );
		if( ff ==  exept.end() )
			::clear((*i));
	}
	Memory.mem_compact();
	clLog( "mem usage after clear build textures surface: %u", Memory.mem_usage() );
}

void	xrLC_GlobalData	::create_rcmodel	(CDB::CollectorPacked& CL)
{
	VERIFY(!_cl_globs._RCAST_Model);
	_cl_globs._RCAST_Model				= xr_new<CDB::MODEL> ();
	_cl_globs._RCAST_Model->build		(CL.getV(),(int)CL.getVS(),CL.getT(),(int)CL.getTS());
}

void		xrLC_GlobalData	::				initialize		()
{
	if (strstr(Core.Params,"-att"))	_gl_linear	= true;
}

base_Face* convert_nax( void* dummy )
{
	return (base_Face*)(dummy);
}

void* convert_nax( base_Face* F )
{
	return (void*)F;
}
 
static xr_vector<Fvector> verts;
static xr_vector<CDB::TRI> tris;
    
void	xrLC_GlobalData::mu_models_calc_materials()
{
	for (u32 m=0; m<mu_models().size(); m++)
		mu_models()[m]->calc_materials();
}

bool			xrLC_GlobalData	::			b_r_vertices	()		
{
	return false;
}

 
template<typename T>
std::pair<u32,u32>	get_id( const xr_vector<xrMU_Model*>& mu_models, const T * v )
{


	u32 face_id = u32(-1);
	struct find
	{
		const T * _v;
		u32& _id;
		find( const T * v, u32& id) : _v(v), _id( id )
		{}
		bool operator () ( const xrMU_Model * m )
		{	
			VERIFY(m);
			u32 id = m->find( _v );
			if( id == u32(-1) )
				return false;
			_id = id;
			return true;
		}
	} f( v, face_id );

	xr_vector<xrMU_Model*> :: const_iterator ii =std::find_if( mu_models.begin(), mu_models.end(), f );
	if( ii == mu_models.end() )
		return std::pair<u32,u32>(u32(-1), u32(-1));
	return std::pair<u32,u32>(u32(ii-mu_models.begin()), face_id );
}

enum serialize_mesh_item_type
{
	smit_plain = u8(0),
	smit_model = u8(1),
	smit_null  = u8(-1)
};

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
void	xrLC_GlobalData::clear_mu_models	()
{	

		clLog( "mem usage before mu_clear %d", Memory.mem_usage() );
		vec_clear(_mu_models);// not clear ogf
		vec_clear(_mu_refs);
		mu_mesh_clear();
		Memory.mem_compact();
		clLog( "mem usage after mu_clear: %d", Memory.mem_usage() );

}
void		xrLC_GlobalData::				clear			()
{
		vec_spetial_clear(_cl_globs._textures );
		_cl_globs._materials.clear();
		_cl_globs._shaders.Unload();
	//	CMemoryWriter					_err_invalid;
	//	b_params						_g_params;
		 

		vec_clear(_g_lightmaps);
		vec_clear(_mu_models);//mem leak
		vec_clear(_mu_refs);
		mu_mesh_clear();
		gl_mesh_clear();
		//VertexPool;
		//FacePool;

	

	//	vecVertex						_g_vertices;
	//	vecFace							_g_faces;
		gl_mesh_clear	();
	    vec_clear		(_g_deflectors);

		//base_lighting					_L_static;
		xr_delete(_cl_globs._RCAST_Model);
 
//		bool							_b_nosun;
//		bool							_gl_linear;
}


void		xrLC_GlobalData::set_faces_indexses		()
{
	//const u32 number = g_faces		().size();
	//for( u32 i=0; i< number; ++i	)
	//	g_faces()[i]->set_index( i );
}
void		xrLC_GlobalData::set_vertices_indexses	()
{
//	const u32 number = g_vertices().size();
//	for( u32 i=0; i< number; ++i	)
//		g_vertices()[i]->set_index( i );
}

