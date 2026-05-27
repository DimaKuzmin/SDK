#pragma once

#include "embree4/rtcore.h"
#include "../XrLCLight/embree_raytracing/EmbreeGeomBuilder.h" 
 
class SceneEmbreeAI
{
	RTCDevice device;
	RTCScene IntelScene;
	RTCGeometry IntelGeometry;
	
 public:
	TriangleContainer static_geom;

 	bool InitedDevice = false;

	void InitializeGeometryNew( );
 	void InitializeEmbree( );
	void ReleaseScene();

	// RayTracing
	float RayTrace(Fvector& P, Fvector& Dir, float R);
};
 