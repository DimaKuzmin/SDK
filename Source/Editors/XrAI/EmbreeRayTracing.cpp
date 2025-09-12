#include "stdafx.h"
#include "EmbreeRayTracing.h"
#include "compiler.h"
 
#pragma comment(lib, "embree4.lib")
#pragma comment(lib, "tbb12.lib")

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	string256 tmp;
	sprintf(tmp, "--- EMBREE: error %d: %s", error, str);
	R_ASSERT(0, tmp);
}

 
void SceneEmbreeAI::BuildRaytraceModel()
{
	// Тут уже будет отфильтровано .
	CTimer t; t.Start();

 	verts_v.swap(build_data.build_verts);
	faces_v.resize(build_data.build_fcnt);

	for (auto Fid = 0; Fid < build_data.build_faces.size(); Fid++)
	{
		auto& FCDB = build_data.build_faces[Fid];
 		faces_v[Fid].point1 = FCDB.verts[0];
		faces_v[Fid].point2 = FCDB.verts[1];
		faces_v[Fid].point3 = FCDB.verts[2];
	}

	// Чистим вектора
	build_data.build_faces.clear();
	build_data.build_faces.shrink_to_fit();

	build_data.build_verts.clear();
	build_data.build_verts.shrink_to_fit();

	build_data.build_fcnt = 0;
	build_data.build_vcnt = 0;

	clMsg("$[Embree] Loading Triangle: %u | vertex: %u", faces_v.size(), verts_v.size() );
	clMsg("$[Embree] Loading Geometry Time: %u ms", t.GetElapsed_ms());
}
 
void SceneEmbreeAI::InitializeGeometryNew(RTCFilterFunctionN filter_fuction)
{
 	BuildRaytraceModel();

	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_LOW);
 	rtcSetGeometryIntersectFilterFunction(IntelGeometry, filter_fuction);

 	rtcSetSharedGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, verts_v.data(), 0, sizeof(Fvector), verts_v.size());
	rtcSetSharedGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,	faces_v.data(), 0, sizeof(Triangle), faces_v.size());
 	rtcCommitGeometry(IntelGeometry);


	// Check need
	IntelScene = rtcNewScene(device);
  	rtcAttachGeometry(IntelScene, IntelGeometry);
 	rtcCommitScene(IntelScene);
}
 
void SceneEmbreeAI::InitializeEmbree(RTCFilterFunctionN filter_fuction)
{
	std::string config;
	bool avx = true, sse = false;
	if (avx = strstr(Core.Params, "-use_avx"))
		config = "threads=8,isa=avx2";
	else if (sse = strstr(Core.Params, "-use_sse"))
		config = "threads=8,isa=sse4.2";
	else
		config = "threads=8,isa=sse2";

	device = rtcNewDevice(config.c_str());
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
	Msg("Intilized Intel Embree v4.1.0 - %s", avx ? "avx" : sse ? "sse" : "default");
	
	// Scene
	InitedDevice = true;
 	InitializeGeometryNew(filter_fuction);
}

void SceneEmbreeAI::ReleaseScene()
{
	InitedDevice = false;

	rtcReleaseGeometry(IntelGeometry);
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}
 

void SceneEmbreeAI::RayTrace(RTCRayHit* rayhit, RayQuaryStructure* context, bool bCull)
{
	if (!InitedDevice)
	{
		Msg("--- Try Ray Trace But Not Initialized Embree!!!");
 		return;
	}
	RTCIntersectArguments args;
 	rtcInitIntersectArguments(&args);
	args.context = &context->context;

 	rtcIntersect1(IntelScene, rayhit, &args);
}
 