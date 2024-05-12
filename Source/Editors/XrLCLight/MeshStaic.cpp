#include "stdafx.h"

#include "xrLC_GlobalData.h"

#include "xrface.h"
#include "vector_clear.h"

typedef poolSS<Vertex,8*1024>	poolVertices;
typedef poolSS<Face,8*1024>		poolFaces;
static poolVertices	_VertexPool;
static poolFaces	_FacePool;

Face* xrLC_GlobalData	::create_face	()		
{
	return _FacePool.create();
}
void xrLC_GlobalData	::destroy_face	(Face* &f)
{
	_FacePool.destroy( f );
}

Vertex* xrLC_GlobalData	::create_vertex	()		
{
	return _VertexPool.create();
}
void xrLC_GlobalData	::destroy_vertex	(Vertex* &f)
{
	_VertexPool.destroy( f );
}




static struct destruct_vertex_not_uregister
{
	static void destruct (Vertex * &v)
	{
		::destroy_vertex( v, false );
	}
} _destruct_vertex_not_uregister;
static struct destruct_face_not_uregister
{
	static void destruct (Face * &f)
	{
		::destroy_face( f, false );
	}
} _destruct_face_not_uregister;
void xrLC_GlobalData	::gl_mesh_clear	()
{
	vec_clear( _g_vertices, _destruct_vertex_not_uregister ); 
	vec_clear( _g_faces, _destruct_face_not_uregister );
	
	_VertexPool.clear();
	_FacePool.clear();
}

 
void	xrLC_GlobalData::clear_mesh		()
{
	
	//R_ASSERT(g_XSplit.empty());
	clLog( "mem usage before clear mesh: %u", Memory.mem_usage() );
	//g_vertices().clear();
	//g_faces().clear();
	//_VertexPool.clear();
	//_FacePool.clear();
	gl_mesh_clear	();
	Memory.mem_compact();
	clLog( "mem usage after clear mesh: %u", Memory.mem_usage() );
}









