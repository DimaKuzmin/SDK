#include "stdafx.h"

#include "xrface.h"
//#include "build.h"
#include "xrdeflector.h"
#include "xrLC_globaldata.h"
#include "serialize.h"
#include "lightmap.h"

volatile u32					dwInvalidFaces;
u32		InvalideFaces()
{
	return dwInvalidFaces;
}

const Shader_xrLC&	base_Face::Shader		()const
{
	VERIFY( inlc_global_data() );
	return shader( dwMaterial, inlc_global_data()->shaders(), inlc_global_data()->materials() );
}

const Shader_xrLC& base_Face::Shader_xrLC()		const
{
	VERIFY(inlc_global_data());
	return shader( dwMaterialGame, inlc_global_data()->shaders(), inlc_global_data()->materials());
}

void			base_Face::CacheOpacity	()
{
	flags.bOpaque				= true;
	VERIFY ( inlc_global_data() );

	b_material& M		= inlc_global_data()->materials()		[dwMaterial];
	b_BuildTexture&	T	= inlc_global_data()->textures()		[M.surfidx];
	if (T.bHasAlpha)	
		flags.bOpaque = false;
	else	
		flags.bOpaque = true;

	if ( !flags.bOpaque && !(T.THM.HasSurface()))	//(0==T.pSurface)//	pSurface was possible deleted
	{
		flags.bOpaque	= true;
		clMsg			("[Prepare Light] Strange face detected... Has alpha without texture...: %s", T.name);
	}
}
static bool do_not_add_to_vector_in_global_data = false;
Face*	Face::read_create( )
{
	do_not_add_to_vector_in_global_data = true;
	Face* f =  inlc_global_data()->create_face();
	do_not_add_to_vector_in_global_data = false;
	
	return f;
}

bool			g_bUnregister = true;

void destroy_vertex( Vertex* &v, bool unregister )
{
	bool tmp_unregister = g_bUnregister;
	g_bUnregister = unregister;
	inlc_global_data()->destroy_vertex( v );
	g_bUnregister = tmp_unregister;
}

void destroy_face( Face* &v, bool unregister )
{
	bool tmp_unregister = g_bUnregister;
	g_bUnregister = unregister;
	inlc_global_data()->destroy_face( v );
	g_bUnregister = tmp_unregister;
}

Tvertex<DataVertex>::Tvertex()
{
	
	VERIFY( inlc_global_data() );
	if( inlc_global_data()->vert_construct_register() )
	{	
 		inlc_global_data()->g_vertices().push_back(this);
	}
	m_adjacents.reserve	(4);
}

template <>
Tvertex<DataVertex>::~Tvertex()
{
	if (g_bUnregister) 
	{
		vecVertexIt F = std::find(inlc_global_data()->g_vertices().begin(), inlc_global_data()->g_vertices().end(), this);
		if (F!=inlc_global_data()->g_vertices().end())
		{
			vecVertex& verts = inlc_global_data()->g_vertices();
			std::swap( *F, *( verts.end()-1 ) );
			verts.pop_back();
		}
		else 
			clMsg("* ERROR: Unregistered VERTEX destroyed");
	}
}

IC Vertex*	Vertex::CreateCopy_NOADJ( vecVertex& vertises_storage ) const
{
	VERIFY( &vertises_storage == &inlc_global_data()->g_vertices() );
	Vertex* V	= inlc_global_data()->create_vertex();
	V->P.set	(P);
	V->N.set	(N);
	V->C		= C;
	return		V;
}

Vertex*	Vertex::read_create( )
{

	return inlc_global_data()->create_vertex();;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
 

template<>
Tface<DataVertex>::Tface()
{
	pDeflector				= 0;
	flags.bSplitted			= false;
	VERIFY( inlc_global_data() );
	if( !do_not_add_to_vector_in_global_data )
	{
		//set_index( inlc_global_data()->g_faces().size() );
		inlc_global_data()->g_faces().push_back		(this);
	}
	sm_group				= u32(-1);
	lmap_layer				= NULL;
}

template<>
Tface<DataVertex>::~Tface()
{
	if (g_bUnregister) 
	{
		vecFaceIt F = std::find(inlc_global_data()->g_faces().begin(), inlc_global_data()->g_faces().end(), this);
		if (F!=inlc_global_data()->g_faces().end())
		{
			vecFace& faces = inlc_global_data()->g_faces();
			std::swap( *F, *( faces.end()-1 ) );
			faces.pop_back();
			//faces.erase(F);
		}
		else clMsg("* ERROR: Unregistered FACE destroyed");
	}
	// Remove 'this' from adjacency info in vertices
	for (int i=0; i<3; ++i)
		v[i]->prep_remove(this);

	lmap_layer				= NULL;
}

//#define VPUSH(a) ((a).x), ((a).y), ((a).z)

template<>
void Face::	Failure		()
{
	dwInvalidFaces			++;

	clMsg		("* ERROR: Invalid face. (A=%f,e0=%f,e1=%f,e2=%f)",
		CalcArea(),
		v[0]->P.distance_to(v[1]->P),
		v[0]->P.distance_to(v[2]->P),
		v[1]->P.distance_to(v[2]->P)
		);
	clMsg		("*        v0[%f,%f,%f], v1[%f,%f,%f], v2[%f,%f,%f]",
		VPUSH(v[0]->P),
		VPUSH(v[1]->P),
		VPUSH(v[2]->P)
		);
	inlc_global_data()->err_invalid().w_fvector3	(v[0]->P);
	inlc_global_data()->err_invalid().w_fvector3	(v[1]->P);
	inlc_global_data()->err_invalid().w_fvector3	(v[2]->P);
}

void	Face::Verify		()
{
	// 1st :: area
	float	_a	= CalcArea();
	if		(!_valid(_a) || (_a<EPS))		{ Failure(); return; }

	// 2nd :: TC0
	Fvector2*	tc			= getTC0();
	float	eps				= .5f / 4096.f;		// half pixel from 4096 texture (0.0001220703125)
	float	e0				= tc[0].distance_to(tc[1]);	
	float	e1				= tc[1].distance_to(tc[2]);
	float	e2				= tc[2].distance_to(tc[0]);
	float	p				= e0+e1+e2;
	if		(!_valid(_a) || (p<eps))		{ Failure(); return; }

	// 3rd :: possibility to calc normal
	CalcNormal				();
	if (!_valid(N))			{ Failure(); return; }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int			affected	= 0;
 
void start_unwarp_recursion()
{
	affected				= 1;
}
 
void Face::OA_Unwarp( CDeflector *D, xr_vector<type_face*>& faces )
{
	if (pDeflector)					return;
	if (!D->OA_Place(this))		    return;
	
	faces.push_back(this);

	// now iterate on all our neigbours
	for (int i=0; i<3; ++i) 
	for (vecFaceIt it=v[i]->m_adjacents.begin(); it!=v[i]->m_adjacents.end(); ++it) 
	{
		affected		+= 1;
		(*it)->OA_Unwarp(D, faces);
	}
}


BOOL	DataFace::RenderEqualTo	(Face *F)
{
	if (F->dwMaterial	!= dwMaterial		)	return FALSE;
 	return TRUE;
}



void	DataFace::AddChannel	(Fvector2 &p1, Fvector2 &p2, Fvector2 &p3) 
{
	_TCF	TC;
	TC.uv[0] = p1;	TC.uv[1] = p2;	TC.uv[2] = p3;
	tc.push_back(TC);
}

BOOL	DataFace::hasImplicitLighting()
{
	if (0==this)								return FALSE;
	if (!Shader().flags.bRendering)				return FALSE;
	VERIFY( inlc_global_data() );
	b_material& M		= inlc_global_data()->materials()		[dwMaterial];
	b_BuildTexture&	T	= inlc_global_data()->textures()		[M.surfidx];
	return (T.THM.flags.test(STextureParams::flImplicitLighted));
}

void	DataFace::	read	(INetReader	&r )
{
	base_Face::read( r );	

	r.r_fvector3( N );			
	r_vector ( r, tc ) ;			
	pDeflector =0 ;
	VERIFY( read_lightmaps );
	read_lightmaps->read( r, lmap_layer );
	sm_group = r.r_u32();

}
void	DataFace::	write	(IWriter	&w )const
{
	base_Face::write( w );
	w.w_fvector3( N );			
	w_vector ( w, tc ) ;			
	VERIFY( write_lightmaps );
	write_lightmaps->write( w, lmap_layer );
	w.w_u32( sm_group );
}
	
void	DataVertex::	read	(INetReader	&r )
{
	base_Vertex::read( r );
}

void	DataVertex::	write	(IWriter	&w )const
{
	base_Vertex::write( w );
}

void Face::	read_vertices		( INetReader	&r )
{
	VERIFY( ::read_vertices );
	::read_vertices->read( r, v[0] );
	::read_vertices->read( r, v[1] );
	::read_vertices->read( r, v[2] );
}
void Face::write_vertices		( IWriter	&w )const
{
	VERIFY( ::write_vertices );
	::write_vertices->write( w, v[0] );
	::write_vertices->write( w, v[1] );
	::write_vertices->write( w, v[2] );
}

void	Face::	read	( INetReader	&r )
{
	DataFace::read( r );
}

void	Face::	write	( IWriter	&w )const
{
	DataFace::write( w );
}




void	Vertex::read		( INetReader	&r )
{
	//	v_faces							m_adjacents; !
	DataVertex::read( r );
}
void	Vertex::write		( IWriter	&w )const
{
	//	v_faces							m_adjacents; !
	DataVertex::write( w );
}

//////////////////////////////////////////////////////////////
void	Vertex::isolate_pool_clear_read		( INetReader	&r )
{
	DataVertex::read( r );
	r_pod_vector( r, m_adjacents );
	for(u32 i= 0; i< m_adjacents.size();++i )
	{
		Face &f = *m_adjacents[i];
		int v_i = -1;
		r_pod( r, v_i );
		R_ASSERT( v_i>=0 );
		R_ASSERT( v_i<3 );
		R_ASSERT( f.vertex( v_i ) == 0 );
		f.raw_set_vertex( v_i, this );
	}
}
void	Vertex::isolate_pool_clear_write	( IWriter	&w )const
{
	DataVertex::write( w );
	w_pod_vector( w, m_adjacents );
	for(u32 i= 0; i< m_adjacents.size();++i )
	{
		Face &f = *m_adjacents[i];
		int v_i = f.VIndex( this );
		R_ASSERT( v_i>=0 );
		R_ASSERT( v_i<3 );
		w_pod( w, v_i );
		f.raw_set_vertex( v_i, 0 );
	}
}

void	Vertex::read_adjacents		( INetReader	&r )
{
	//VERIFY()
}
void	Vertex::write_adjacents		( IWriter	&w )const
{

}

///////////////////////////////////////////////////////////////
