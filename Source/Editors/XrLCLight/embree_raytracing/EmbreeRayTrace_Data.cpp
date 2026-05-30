#include "stdafx.h"
#include "xrDeflector.h"
#include "R_light.h"
#include "light_point.h"
#include "base_lighting.h"
#include "xrLC_GlobalData.h"

#include "EmbreeRayTrace.h"
#include "xrMU_Model_Reference.h"
#include "xrMU_Model.h"

#include "base_face.h"

// Для Загрузки Геометрии
extern CompilersMode gCompilerMode;

void SetRay1(RTCRay& rayhit, Fvector& pos, Fvector& dir, float near_, float range)
{
	rayhit.dir_x = dir.x;
	rayhit.dir_y = dir.y;
	rayhit.dir_z = dir.z;
	rayhit.org_x = pos.x;
	rayhit.org_y = pos.y;
	rayhit.org_z = pos.z;
	rayhit.tnear = near_;
	rayhit.tfar = range;
	rayhit.mask = (unsigned int)(-1);
	rayhit.flags = 0;
}

void SetRay1(RTCRayHit& rayhit, Fvector& pos, Fvector& dir, float near_, float range)
{
	rayhit.ray.dir_x = dir.x;
	rayhit.ray.dir_y = dir.y;
	rayhit.ray.dir_z = dir.z;
	rayhit.ray.org_x = pos.x;
	rayhit.ray.org_y = pos.y;
	rayhit.ray.org_z = pos.z;
	rayhit.ray.tnear = near_;
	rayhit.ray.tfar = range;
	rayhit.ray.mask = (unsigned int)(-1);
	rayhit.ray.flags = 0;

	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instPrimID[0] = RTC_INVALID_GEOMETRY_ID;
}

// OFF PACKED PROCESSING
void GetEmbreeDeviceProperty(const char* msg, RTCDevice& device, RTCDeviceProperty prop)
{
	Msg(" - EmbreeDevProp: %s : %llu", msg, rtcGetDeviceProperty(device, prop));
}


IC bool	FaceEqual__(Face& F1, Face& F2)
{
	// Test for 6 variations
	if ((F1.v[0] == F2.v[0]) && (F1.v[1] == F2.v[1]) && (F1.v[2] == F2.v[2])) return true;
	if ((F1.v[0] == F2.v[0]) && (F1.v[2] == F2.v[1]) && (F1.v[1] == F2.v[2])) return true;
	if ((F1.v[2] == F2.v[0]) && (F1.v[0] == F2.v[1]) && (F1.v[1] == F2.v[2])) return true;
	if ((F1.v[2] == F2.v[0]) && (F1.v[1] == F2.v[1]) && (F1.v[0] == F2.v[2])) return true;
	if ((F1.v[1] == F2.v[0]) && (F1.v[0] == F2.v[1]) && (F1.v[2] == F2.v[2])) return true;
	if ((F1.v[1] == F2.v[0]) && (F1.v[2] == F2.v[1]) && (F1.v[0] == F2.v[2])) return true;
	return false;
}

void EmbreeRayTraceModel::BuildModel(xr_vector<FaceDataEmbree>& faces)
{
	static_geom.ClearAll();
	static_geom_transp.ClearAll();

	int IndexFace = 0, IndexFaceTransp = 0;
	for (auto& Fe : faces)
	{
		Face* F = (Face*)Fe.ptr;

		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T  = inlc_global_data()->textures()[M.surfidx];

		bool isOpcue = F->flags.bOpaque || T.pSurface.Empty() || !T.bHasAlpha;
		auto& geom_buff = isOpcue ? static_geom : static_geom_transp;
		geom_buff.AddFaceRaw(F, Fe.v1, Fe.v2, Fe.v3);

		if (isOpcue)  IndexFace++;
		if (!isOpcue) IndexFaceTransp++;
	}

	static_geom.RemoveDublicates();
 	static_geom_transp.RemoveDublicates();
}

void EmbreeRayTraceModel::BuildRaytraceModel()
{
	static_geom.ClearAll();
	static_geom_transp.ClearAll();

	CTimer t;	t.Start();
	Status("[RcastModel] Capturing Faces...");
	for (auto F : lc_global_data()->g_faces())
	{
		const Shader_xrLC& SH = F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)	continue;

		b_material& M = inlc_global_data()->materials()[F->dwMaterial];
		b_texture& T = inlc_global_data()->textures()[M.surfidx];
		if (F->flags.bOpaque || T.pSurface.Empty() || !T.bHasAlpha)
			static_geom.AddFaceRaw(F, F->v[0]->P, F->v[1]->P, F->v[2]->P);
		else
			static_geom_transp.AddFaceRaw(F, F->v[0]->P, F->v[1]->P, F->v[2]->P);
	}


	for (auto ref : lc_global_data()->mu_refs())
	{
		xr_vector<FaceDataEmbree> temp_buffer;
		ref->export_cform_rcast_new(temp_buffer);

		for (auto& FaceIntel : temp_buffer)
		{
			Face* F = (Face*)FaceIntel.ptr;
			b_material& M = inlc_global_data()->materials()[F->dwMaterial];
			b_texture& T = inlc_global_data()->textures()[M.surfidx];
			if (F->flags.bOpaque || T.pSurface.Empty() || !T.bHasAlpha)
				static_geom.AddFaceRaw(F, FaceIntel.v1, FaceIntel.v2, FaceIntel.v3);
			else
				static_geom_transp.AddFaceRaw(F, FaceIntel.v1, FaceIntel.v2, FaceIntel.v3);
		}

	}
	Status("[RcastModel] Capturing Faces [%u ms]", t.GetElapsed_ms());

	static_geom_transp.RemoveDublicates();
	static_geom.RemoveDublicates();
}
