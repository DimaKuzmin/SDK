#include "stdafx.h"
#include "xrCDB_Embree.h"

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	string256 tmp;
	sprintf(tmp, "--- EMBREE: error %d: %s", error, str);
	R_ASSERT(0, tmp);
}

/*
  RTC_DEVICE_PROPERTY_VERSION       = 0,
  RTC_DEVICE_PROPERTY_VERSION_MAJOR = 1,
  RTC_DEVICE_PROPERTY_VERSION_MINOR = 2,
  RTC_DEVICE_PROPERTY_VERSION_PATCH = 3,

  RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED  = 32,
  RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED  = 33,
  RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED = 34,

  RTC_DEVICE_PROPERTY_BACKFACE_CULLING_SPHERES_ENABLED = 62,
  RTC_DEVICE_PROPERTY_BACKFACE_CULLING_CURVES_ENABLED = 63,
  RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED          = 64,
  RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED    = 65,
  RTC_DEVICE_PROPERTY_FILTER_FUNCTION_SUPPORTED   = 66,
  RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED = 67,
  RTC_DEVICE_PROPERTY_COMPACT_POLYS_ENABLED       = 68,

  RTC_DEVICE_PROPERTY_TRIANGLE_GEOMETRY_SUPPORTED    = 96,
  RTC_DEVICE_PROPERTY_QUAD_GEOMETRY_SUPPORTED        = 97,
  RTC_DEVICE_PROPERTY_SUBDIVISION_GEOMETRY_SUPPORTED = 98,
  RTC_DEVICE_PROPERTY_CURVE_GEOMETRY_SUPPORTED       = 99,
  RTC_DEVICE_PROPERTY_USER_GEOMETRY_SUPPORTED        = 100,
  RTC_DEVICE_PROPERTY_POINT_GEOMETRY_SUPPORTED       = 101,

  RTC_DEVICE_PROPERTY_TASKING_SYSTEM        = 128,
  RTC_DEVICE_PROPERTY_JOIN_COMMIT_SUPPORTED = 129,
  RTC_DEVICE_PROPERTY_PARALLEL_COMMIT_SUPPORTED = 130
*/

 
void GetEmbreeDeviceProperty(LPCSTR msg, RTCDevice& device, RTCDeviceProperty prop)
{
	Msg("EmbreeDevProp: %s : %llu", msg, rtcGetDeviceProperty(device, prop));
}
 
void CDB::Embree::SceneEmbree::InitGeometry(CDB::TRI* tris_buff, u32 tris_cnt, Fvector* verts_buff, u32 verts_cnt, RTCFilterFunctionN filter_fuction)
{
	std::string config;
	bool avx = false, sse = false;
	if (avx = strstr(Core.Params, "-use_avx"))
		config = "threads=8,isa=avx2";
	else if (sse = strstr(Core.Params, "-use_sse"))
		config = "threads=8,isa=sse4.2";
	else
		config = "threads=8,isa=sse2";

	device = rtcNewDevice(config.c_str());
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED", device, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED", device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED", device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED);
	

	
	Msg("Intilized Intel Embree v4.1.0 - %s", avx ? "avx" : sse ? "sse" : "default");
	// Scene
	IntelScene = rtcNewScene(device); 

	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

	// Создание сцены и добавление геометрии
	 
	vertices = (VertexEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VertexEmbree), verts_cnt);
	triangles = (TriEmbree*)rtcSetNewGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(TriEmbree), tris_cnt);

	v_cnt = verts_cnt;
	t_cnt = tris_cnt;
 
	for (int i = 0; i < verts_cnt; i++)
	{
		Fvector verts = verts_buff[i];
		vertices[i] = VertexEmbree{ verts.x, verts.y, verts.z };
	}
 
	for (int i = 0; i < tris_cnt; i++)
	{
		CDB::TRI tri = tris_buff[i];
		triangles[i] = TriEmbree{ tri.verts[0], tri.verts[1], tri.verts[2] };
	}

	// Check need
	rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_REFIT);
	rtcCommitGeometry(IntelGeometry);
	rtcAttachGeometry(IntelScene, IntelGeometry);

	rtcCommitScene(IntelScene);
	rtcSetGeometryIntersectFilterFunction(IntelGeometry, filter_fuction);

	InitedDevice = true;
}

void CDB::Embree::SceneEmbree::ReleaseScene()
{
	rtcReleaseGeometry(IntelGeometry);
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);

	InitedDevice = false;
}
 

void CDB::Embree::SceneEmbree::RayTrace(RTCRayHit* rayhit, RayQuaryStructure* context, RTCFilterFunctionN filter_fuction, bool bCull)
{
	if (!InitedDevice)
	{
		Msg("--- Try Ray Trace But Not Initialized Embree!!!");

		return;
	}

	
	RTCRayQueryContext ctxt;
	rtcInitRayQueryContext(&ctxt);

	context->context = ctxt;
	context->SceneEmbree = this;

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);

	args.context = &context->context;
	args.filter = filter_fuction;
 	args.flags = (RTCRayQueryFlags)(RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER | RTC_RAY_QUERY_FLAG_COHERENT );

	rtcIntersect1(IntelScene, rayhit, &args);
}
 