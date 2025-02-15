#include "stdafx.h"
//#include "igame_level.h"

#include "xr_area.h"
#include "../xrengine/xr_object.h"
#include "../xrengine/xrLevel.h"
#include "../xrengine/xr_collide_form.h"
//#include "../xrsound/sound.h"
//#include "x_ray.h"
//#include "GameFont.h"


using namespace	collide;

//----------------------------------------------------------------------
// Class	: CObjectSpace
// Purpose	: stores space slots
//----------------------------------------------------------------------
CObjectSpace::CObjectSpace	( ):
	xrc()
#ifdef DEBUG
	,m_pRender(0)
#endif
{
#ifdef DEBUG
	if( RenderFactory )	
		m_pRender = CNEW(FactoryPtr<IObjectSpaceRender>)() ;
#endif
	m_BoundingVolume.invalidate	();
}
//----------------------------------------------------------------------
CObjectSpace::~CObjectSpace	( )
{
	//moved to ~IGameLevel
#ifdef DEBUG
	CDELETE(m_pRender);
#endif
}
//----------------------------------------------------------------------

//----------------------------------------------------------------------
int CObjectSpace::GetNearest		( xr_vector<ISpatial*>& q_spatial, xr_vector<CObject*>&	q_nearest, const Fvector &point, float range, CObject* ignore_object )
{
	q_spatial.clear_not_free		( );
	// Query objects
	q_nearest.clear_not_free		( );
	Fsphere				Q;	Q.set	(point,range);
	Fvector				B;	B.set	(range,range,range);
	g_SpatialSpace->q_box(q_spatial,0,STYPE_COLLIDEABLE,point,B);

	// Iterate
	xr_vector<ISpatial*>::iterator	it	= q_spatial.begin	();
	xr_vector<ISpatial*>::iterator	end	= q_spatial.end		();
	for (; it!=end; it++)		{
		CObject* O				= (*it)->dcast_CObject		();
		if (0==O)				continue;
		if (O==ignore_object)	continue;
		Fsphere mS				= { O->spatial.sphere.P, O->spatial.sphere.R	};
		if (Q.intersect(mS))	q_nearest.push_back(O);
	}

	return q_nearest.size();
}

//----------------------------------------------------------------------
IC int	CObjectSpace::GetNearest	( xr_vector<CObject*>&	q_nearest, const Fvector &point, float range, CObject* ignore_object )
{
	return							(
		GetNearest(
			r_spatial,
			q_nearest,
			point,
			range,
			ignore_object
		)
	);
}

//----------------------------------------------------------------------
IC int   CObjectSpace::GetNearest( xr_vector<CObject*>&	q_nearest, ICollisionForm* obj, float range)
{
	CObject*	O		= obj->Owner	();
	return				GetNearest( q_nearest, O->spatial.sphere.P, range + O->spatial.sphere.R, O );
}

//----------------------------------------------------------------------


void CObjectSpace::Load	( CDB::build_callback build_callback )
{
	Load("$level$","level.cform", build_callback);
}
void	CObjectSpace::		Load				(  LPCSTR path, LPCSTR fname, CDB::build_callback build_callback  )
{
	IReader *F					= FS.r_open	(path, fname);
	R_ASSERT					(F);
	Load( F, build_callback );
}
void	CObjectSpace::	Load				(  IReader* F, CDB::build_callback build_callback  )


{
	hdrCFORM					H;
	F->r						(&H,sizeof(hdrCFORM));
	Fvector*	verts			= (Fvector*)F->pointer();
	xr_vector< CDB::TRI> tris(H.facecount);
	{
		u8* tris_pointer = (u8*)(verts + H.vertcount);
		for (size_t i = 0; i < H.facecount; i++)
		{
			memcpy(&tris[i], tris_pointer, CDB::TRI::Size());
			tris_pointer += CDB::TRI::Size();
		}
		Create(verts, tris.data(), H, build_callback);

	}
	FS.r_close					(F);
}

void			CObjectSpace::Create				(  Fvector*	verts, CDB::TRI* tris, const hdrCFORM &H, CDB::build_callback build_callback  )
{
	R_ASSERT							(CFORM_CURRENT_VERSION==H.version);
	Static.build						( verts, H.vertcount, tris, H.facecount, build_callback );
	m_BoundingVolume.set				(H.aabb);
	g_SpatialSpace->initialize			(m_BoundingVolume);
	g_SpatialSpacePhysic->initialize	(m_BoundingVolume);
}

//----------------------------------------------------------------------
#ifdef DEBUG
void CObjectSpace::dbgRender()
{
	(*m_pRender)->dbgRender();
}
#endif
