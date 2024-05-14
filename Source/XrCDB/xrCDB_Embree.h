#pragma once

#include "embree4/rtcore.h"
#include "xrCDB.h"
 
 
struct XRCDB_API RayQuaryStructure
{
	RTCRayQueryContext context;
	void* SceneEmbree = 0;

	RTCRayHit hits[1024];
	int count = 0;

	// RayTrace Tris
	void* skip = 0;
	void* light = 0;


	float energy = 1.0f;
};

struct XRCDB_API VertexEmbree { float x, y, z; };
struct XRCDB_API TriEmbree { uint32_t point1, point2, point3; };

class XRCDB_API SceneEmbree
{
	RTCDevice device;
	RTCScene IntelScene;
	RTCGeometry IntelGeometry;

	VertexEmbree* vertices;
	TriEmbree* triangles;
	u32 v_cnt = 0;
	u32 t_cnt = 0;



public:
	bool InitedDevice = false;

	void InitGeometry(CDB::TRI* tri, u32 tris_cnt, Fvector* verts, u32 verts_cnt, RTCFilterFunctionN filter_fuction);

	void ReleaseScene();



	void RayTrace(RTCRayHit* rayhit, RayQuaryStructure* context, RTCFilterFunctionN filter_fuction, bool bCull);

};
 