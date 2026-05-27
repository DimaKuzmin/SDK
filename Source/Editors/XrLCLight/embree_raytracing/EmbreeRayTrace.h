#pragma once

#include "R_light.h"
#include "base_lighting.h"
#include "base_color.h"
#include "../xrCDB/xrCDB.h"

#include "xrFace.h"
#include <embree4/rtcore.h>

#include "EmbreeGeomBuilder.h"

// ВАЖНЫЙ ПАРАМЕТР TNEAR Для пересечения с водой
void SetRay1(RTCRay& rayhit, Fvector& pos, Fvector& dir, float near_, float range);
void SetRay1(RTCRayHit& rayhit, Fvector& pos, Fvector& dir, float near_, float range);

// Vertex, Tri Buffers
static RTCDevice	EmbreeDevice = nullptr;
static bool			isDeviceInitialized = false;

const char* GetDeviceConfig();
void InitializeEmbreeDevice();



class EmbreeRayTraceModel
{
protected:
	RTCSceneFlags	scene_flags = RTC_SCENE_FLAG_NONE;
	RTCBuildQuality scene_quality = RTC_BUILD_QUALITY_LOW;

	RTCScene	IntelScene = nullptr;
	RTCGeometry IntelGeometryNormal = nullptr;
	RTCGeometry IntelGeometryTransp = nullptr;

	/** NORMAL GEOM **/
	TriangleContainer			static_geom;
	TriangleContainer			static_geom_transp;


	void RemoveGeometry();
	void CommitScene();

	void BuildModel(xr_vector<FaceDataEmbree>& faces);
 	void BuildRaytraceModel();

public:
 	// Loading 
	float RaytraceEmbreeProcess(Fvector& P, Fvector& N, float range, void* skip);
	void  InitializeGeometry();		// Rcast-model
	void  InitializeGeometry_Model(xr_vector<FaceDataEmbree>& faces); // Single-Models (xrMU-Model)

	void  IntelEmbereUnloadAll();

	// Details Loading 
	RTCScene	IntelSceneDetails = nullptr;
	RTCGeometry IntelGeometryDetails = nullptr;
	float RaytraceEmbreeDetails(Fvector& P, Fvector& N, float range);
	void InitEmbreeDetails(TriangleContainer& data);
};

extern EmbreeRayTraceModel EmbreeMain;

void GetEmbreeDeviceProperty(const char* msg, RTCDevice& device, RTCDeviceProperty prop);