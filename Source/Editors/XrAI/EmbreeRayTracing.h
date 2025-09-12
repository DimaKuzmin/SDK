#pragma once

#include "embree4/rtcore.h"
 
struct RayQuaryStructure
{
	RTCRayQueryContext context;
 
	// RayTrace Tris
	void* skip = 0;
 	float energy = 1.0f;
};
  
struct BuildData
{
	xr_vector<CDB::TRI> build_faces;
	xr_vector<Fvector>  build_verts;

	u32		  build_vcnt;
	u32		  build_fcnt;
};

struct Triangle
{
	u32 point1, point2, point3;

	CDB::TRI Get()
	{
		CDB::TRI tri;
		tri.verts[0] = point1;
		tri.verts[1] = point2;
		tri.verts[2] = point3;
		return tri;
	}

	void Set(CDB::TRI& T)
	{
		point1 = T.verts[0];
		point2 = T.verts[1];
		point3 = T.verts[2];
	}
};

class SceneEmbreeAI
{
	RTCDevice device;
	RTCScene IntelScene;
	RTCGeometry IntelGeometry;

	xr_vector<Fvector>				verts_v;
	xr_vector<Triangle>				faces_v;
 
public:
	BuildData build_data;
	bool InitedDevice = false;

	void InitializeGeometryNew(RTCFilterFunctionN filter_fuction);
 	void InitializeEmbree(RTCFilterFunctionN filter_fuction);
	void ReleaseScene();

	void BuildRaytraceModel();

	// RayTracing
	void RayTrace(RTCRayHit* rayhit, RayQuaryStructure* context, bool bCull);

};
 