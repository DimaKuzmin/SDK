#include "stdafx.h"
#include "EmbreeRayTrace.h"
#include "../xrCDB/xrCDB.h"
#include "global_calculation_data.h"
#include "xrLC_GlobalData.h"

extern global_claculation_data	gl_data;

struct RayQueryContext
{
	RTCRayQueryContext context;
	Fvector B;

	void* skip = 0;
	float energy = 1.0f;
};

struct UserGeomData
{
	xr_vector<FaceDataEmbree*> Faces;
};

bool CalculateEnergy(FaceDataEmbree& F, Fvector& B, float& energy, float hu, float hv)
{
 	b_material& M	= gl_data.g_materials[F.dwMaterial];
	b_texture& T	= gl_data.g_textures[M.surfidx];

	if (!T.bHasAlpha)
		return false;

	if (T.pSurface.Empty())
	{
		T.bHasAlpha = false;
		return false;
	}


	// barycentrics (без Fvector, сразу в скаляры)
	float Barry0 = 1.0f - hu - hv;

	// UV сразу float
	const Fvector2* cuv = F.getTC0();
	float u = cuv[0].x * Barry0 + cuv[1].x * hu + cuv[2].x * hv;
	float v = cuv[0].y * Barry0 + cuv[1].y * hu + cuv[2].y * hv;

	int U = (int)floor(u * float(T.dwWidth) + .5f);
	int V = (int)floor(v * float(T.dwHeight) + .5f);
	U %= T.dwWidth;		if (U < 0) U += T.dwWidth;
	V %= T.dwHeight;	if (V < 0) V += T.dwHeight;

	// fetch pixel
	const uint32_t* raw = static_cast<const uint32_t*>(*T.pSurface);
	uint32_t pixel		= raw[V * T.dwWidth + U];
	uint32_t pixel_a	= (pixel >> 24) & 0xFF;

	// LUT вместо деления и sqr
	float a = float(pixel_a) / 255.f;
	float opacity = 1.f - a * a;
	energy *= opacity;
	if (energy < 0.015f)
		return false;
 
	return true;
}

ICF void FilterRaytraceD(const struct RTCFilterFunctionNArguments* args)
{
	RayQueryContext* ctxt = (RayQueryContext*)args->context;
	RTCHit* hit = (RTCHit*)args->hit;
	RTCRay* ray = (RTCRay*)args->ray;

	auto UD = (UserGeomData*) args->geometryUserPtr;
	auto F = UD->Faces[hit->primID];
	if (!CalculateEnergy(*F, ctxt->B, ctxt->energy, hit->u, hit->v))
	{
		ctxt->energy = 0;
		args->valid[0] = -1; // Остановится
		return;
	}

	args->valid[0] = 0;		 // Продолжить
}

float EmbreeRayTraceModel::RaytraceEmbreeDetails(Fvector& P, Fvector& N, float range)
{
	RayQueryContext data_hits;
	data_hits.skip = 0;
	data_hits.energy = 1.0f;

	RTCRayHit rayhit;
	SetRay1(rayhit, P, N, 0.f, range);

	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);

	data_hits.context = context;
	args.context = &data_hits.context;
	rtcIntersect1(IntelSceneDetails, &rayhit, &args);

	return data_hits.energy;
}

// хм почемуто не хочет с другого места работать 
void errors_embree_det(void* userPtr, enum RTCError code, const char* str)
{
	R_ASSERT2(false, str);
}

RTCDevice DeviceDetails = nullptr;
void EmbreeRayTraceModel::InitEmbreeDetails(TriangleContainer& data)
{
	DeviceDetails = rtcNewDevice(GetDeviceConfig());;
	rtcSetDeviceErrorFunction(DeviceDetails, &errors_embree_det, nullptr);

	// Загрузка Геометрии
	static_geom = data;

	IntelGeometryDetails = rtcNewGeometry(DeviceDetails, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(IntelGeometryDetails, RTCBuildQuality::RTC_BUILD_QUALITY_LOW);
	rtcSetGeometryOccludedFilterFunction(IntelGeometryDetails, &FilterRaytraceD);

	rtcSetSharedGeometryBuffer(IntelGeometryDetails, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, static_geom.vertex().data(), 0, sizeof(Fvector), static_geom.vertex().size());
	rtcSetSharedGeometryBuffer(IntelGeometryDetails, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, static_geom.faces().data(), 0, sizeof(Triangle), static_geom.faces().size());
	
	UserGeomData* Udata = xr_new< UserGeomData>();
 	for (auto& VFace : data.UD())
	{
		Udata->Faces.push_back((FaceDataEmbree*) VFace);
	}

	rtcSetGeometryUserData(IntelGeometryDetails, Udata);
	
	rtcCommitGeometry(IntelGeometryDetails);

	clMsg("Loading Embree : verts[%u] faces[%u]", static_geom.vertex_cnt(), static_geom.faces_cnt());


	IntelSceneDetails = rtcNewScene(DeviceDetails);
	rtcSetSceneFlags(IntelSceneDetails, scene_flags);
	rtcAttachGeometryByID(IntelSceneDetails, IntelGeometryDetails, 0);
	rtcCommitScene(IntelSceneDetails);

	clMsg("Scene Create");
}
