#pragma once
#include "../Public/shader_xrlc.h"

#include "tcf.h"
#include "base_face.h"

#include "MeshStructure.h"
#include <memory>
 
#pragma pack(push,4)

struct DataFace;
class  CLightmap;

struct  DataVertex;
typedef	Tvertex< DataVertex> Vertex;

typedef std::pair<Vertex*, Vertex *>	PAIR_VV;


struct  XRLC_LIGHT_API DataVertex	: public base_Vertex
{
public:
 	typedef		DataFace			DataFaceType;

	bool similar(Tvertex<DataVertex>& V, float eps)
	{
		return P.similar(V.P, eps);
	}
 
	DataVertex				(){};
	virtual		~DataVertex				(){};
};

 
typedef	 Tface<DataVertex>  Face;


struct XRLC_LIGHT_API DataFace	: public base_Face
{
public:

 	Fvector					N;				// face normal
 	svector<_TCF,2>			tc;				// TC

	void*					pDeflector;		// does the face has LM-UV map?
	CLightmap*				lmap_layer;
	u32						sm_group;
	virtual Fvector2*		getTC0			( ) { return tc[0].uv; }


	BOOL		RenderEqualTo		( Face *F );

	void		AddChannel			( Fvector2 p1, Fvector2 p2, Fvector2 p3 ); 
	BOOL		hasImplicitLighting	();

	DataFace(){};
	virtual ~DataFace(){};
};

#include		"xrUVpoint.h"
 
 
extern XRLC_LIGHT_API bool						g_bUnregister;

#pragma pack(pop)
 
extern void start_unwarp_recursion	();
extern void destroy_vertex			( Vertex* &v, bool unregister );
extern void destroy_face			( Face* &v, bool unregister );

extern void FromBarry(Face* F, Fvector& wP, Fvector& wN, Fvector& B);
extern void FromBarryNormalized(Face* F, Fvector& wP, Fvector& wN, Fvector& B);