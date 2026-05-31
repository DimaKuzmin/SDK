#include "stdafx.h"
#include "EmbreeRayTracing.h"
#include "compiler.h"

struct RayQuaryUserData : RTCRayQueryContext
{
 	// RayTrace Tris
	void* skip = 0;
	float energy = 1.0f;
};

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
	string256 tmp;
	sprintf(tmp, "--- EMBREE: error %d: %s", error, str);
	R_ASSERT(0, tmp);
}

SceneEmbreeAI			 SceneEmbreeInterface;
 
void filter_embree_function(const struct RTCFilterFunctionNArguments* args)
{
	RayQuaryUserData* ctxt = (RayQuaryUserData*)	args->context;

	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;
	args->valid[0] = 0;

	if (!ctxt)		return;

	auto& F				= g_embree_faces[hit->primID];
	b_material& M	    = g_materials[F.dwMaterial];
	b_texture& T	    = (*g_textures)[M.surfidx];
	Shader_xrLCVec& LIB = g_shaders_xrlc->Library();

	if (M.shader_xrlc >= LIB.size())			// Hack 0
	{
		ctxt->energy = 0;
		args->valid[0] = -1;
		return;
	}

	if (F.bOpaque)
	{
		ctxt->energy = 0;
		args->valid[0] = -1;
		return;
	}

 	Shader_xrLC& SH = LIB[M.shader_xrlc];
	if (!SH.flags.bLIGHT_CastShadow)
	{
		ctxt->energy = 0;
		args->valid[0] = -1;
		return;
	}

	if (T.pSurface.Empty())
		T.bHasAlpha = FALSE;

	if (!T.bHasAlpha)
	{
		args->valid[0] = -1;
		ctxt->energy = 0;
		return;
	}
 
	// barycentric coords
	// note: W,U,V order
 	float FromBary = (1.0f - hit->u - hit->v);

	// calc UV
	Fvector2* cuv = F.getTC0();
 	float u = cuv[0].x * FromBary + cuv[1].x * hit->u + cuv[2].x * hit->v;
	float v = cuv[0].y * FromBary + cuv[1].y * hit->u + cuv[2].y * hit->v;

	int U = iFloor(u * float(T.dwWidth) + .5f);
	int V = iFloor(v * float(T.dwHeight) + .5f);
	U %= T.dwWidth;		if (U < 0) U += T.dwWidth;
	V %= T.dwHeight;	if (V < 0) V += T.dwHeight;

	u32 pixel = ((u32*)*T.pSurface)[V * T.dwWidth + U];
	u32 pixel_a = color_get_A(pixel);
	float opac = 1.f - float(pixel_a) / 255.f;
	ctxt->energy *= opac;

}
 
float SceneEmbreeAI::RayTrace(Fvector& P, Fvector& D, float R)
{
	RayQuaryUserData data;

	RTCOccludedArguments args;
	rtcInitOccludedArguments(&args);
	args.context = &data;
	rtcInitRayQueryContext(&data);

	// 1. Check cached polygon
	RTCRay ray;
	ray.tfar = R;
	ray.tnear = 0.01f;
 	ray.org_x = P.x;
	ray.org_y = P.y;
	ray.org_z = P.z;
 	ray.dir_x = D.x;
	ray.dir_y = D.y;
	ray.dir_z = D.z;

	data.energy = 1;
	rtcOccluded1(IntelScene, &ray, &args);
 	return data.energy;
}
 
void SceneEmbreeAI::InitializeGeometryNew( )
{
	static_geom.ClearAll();
 	for (auto & F : g_embree_faces)
		static_geom.AddFaceRaw(nullptr, F.v1, F.v2, F.v3);
	static_geom.RemoveDublicates();

	Msg("Static Geom Vert{%u} Tris{%u}", static_geom.vertex_cnt(), static_geom.faces_cnt());

 
	IntelGeometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometry, RTC_BUILD_QUALITY_LOW);
 	rtcSetGeometryOccludedFilterFunction(IntelGeometry, filter_embree_function);

 	rtcSetSharedGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, static_geom.vertex().data(), 0, sizeof(Fvector), static_geom.vertex_cnt());
	rtcSetSharedGeometryBuffer(IntelGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, static_geom.faces().data(), 0, sizeof(Triangle), static_geom.faces_cnt());
 	rtcCommitGeometry(IntelGeometry);


	// Check need
	IntelScene = rtcNewScene(device);
  	rtcAttachGeometry(IntelScene, IntelGeometry);
 	rtcCommitScene(IntelScene);
}
 
void SceneEmbreeAI::InitializeEmbree()
{
	std::string config;
	bool avx = gCompilerMode.use_avx2;
	if (avx)
		config = "threads=8,isa=avx2";
	else 
		config = "threads=8,isa=sse4.2";
 
 	device = rtcNewDevice(config.c_str());
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
	Msg("Intilized Intel Embree v4.1.0 - %s", avx ? "avx" : "sse");
	
	// Scene
	InitedDevice = true;
 	InitializeGeometryNew();
}

void SceneEmbreeAI::ReleaseScene()
{
	InitedDevice = false;

	rtcReleaseGeometry(IntelGeometry);
	rtcReleaseScene(IntelScene);
	rtcReleaseDevice(device);
}
 
