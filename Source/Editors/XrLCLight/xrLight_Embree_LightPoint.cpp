#include "stdafx.h"
#include "xrDeflector.h"
#include "R_light.h"
#include "light_point.h"
#include "base_lighting.h"
#include "xrLC_GlobalData.h"

#include "xrLight_Embree.h"

void Embree::VertexEmbree::Set(Fvector& vertex)
{
	x = vertex.x;
	y = vertex.y;
	z = vertex.z;
}

Fvector Embree::VertexEmbree::Get()
{
	Fvector vertex;
	x = vertex.x;
	y = vertex.y;
	z = vertex.z;
	return vertex;
}

void Embree::TriEmbree::SetVertexes(CDB::TRI& triangle, Fvector* verts, VertexEmbree* emb_verts, size_t& last_index)
{
	point1 = last_index;
	point2 = last_index + 1;
	point3 = last_index + 2;

	int v1 = triangle.verts[0];
	int v2 = triangle.verts[1];
	int v3 = triangle.verts[2];


	emb_verts[last_index].Set(verts[v1]);
	emb_verts[last_index + 1].Set(verts[v2]);
	emb_verts[last_index + 2].Set(verts[v3]);

	last_index += 3;
}

void Embree::TriEmbree::SetVertexes_new(FaceDataIntel& data, VertexEmbree* emb_verts, size_t& last_index)
{
	point1 = last_index;
	point2 = last_index + 1;
	point3 = last_index + 2;
 	emb_verts[last_index].Set(data.v1);
	emb_verts[last_index + 1].Set(data.v2);
	emb_verts[last_index + 2].Set(data.v3);
	last_index += 3;
}

void Embree::TriEmbree::SetVertexes_fast(Fvector* Vs, VertexEmbree* emb_verts, std::atomic<size_t> & last_index)
{
	size_t INDEX = last_index.load();

	point1 = INDEX;
	point2 = INDEX + 1;
	point3 = INDEX + 2;
	emb_verts[INDEX].Set(Vs[0]);
	emb_verts[INDEX + 1].Set(Vs[1]);
	emb_verts[INDEX + 2].Set(Vs[2]);

 	last_index.fetch_add(3);
}


void Embree::SetRay1(RTCRay& rayhit, Fvector& pos, Fvector& dir, float near_, float range)
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

void Embree::SetRay1(RTCRayHit& rayhit, Fvector& pos, Fvector& dir, float near_, float range)
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
}

// OFF PACKED PROCESSING
void GetEmbreeDeviceProperty(LPCSTR msg, RTCDevice& device, RTCDeviceProperty prop)
{
	clMsg("EmbreeDevProp: %s : %llu", msg, rtcGetDeviceProperty(device, prop));
}

void Embree::errorFunction(void* userPtr, RTCError error, const char* str)
{
	clMsg("error %d: %s", error, str);
	DebugBreak();
}


void Embree::IntelEmbreeSettings(RTCDevice& device, bool avx, bool sse)
{
 	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED", device, RTC_DEVICE_PROPERTY_RAY_MASK_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED", device, RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED", device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED", device, RTC_DEVICE_PROPERTY_IGNORE_INVALID_RAYS_ENABLED);

	GetEmbreeDeviceProperty("RTC_DEVICE_PROPERTY_TASKING_SYSTEM", device, RTC_DEVICE_PROPERTY_TASKING_SYSTEM);
}  

#include "xrFace.h"
#include "xrMU_Model_Reference.h"
IC bool				FaceEqual(Face& F1, Face& F2)
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


void Embree::GetGlobalData(bool isTransp, bool isCalculate, std::atomic<size_t>& counts_faces, Embree::VertexEmbree* verts_embree, Embree::TriEmbree* faces_embree, xr_vector<void*>* dummy)
{
	auto& Faces = lc_global_data()->g_faces();

	struct FaceAttached
	{
		Face* faces[36];
		Face* F;
		int used_faces;
	};

	xr_vector<Face*> faces;

	thread_local xr_vector<Face*>			adjacent_vec(6 * 2 * 3);
	std::atomic<size_t> count_verts = 0;
	for (auto F : Faces)
	{
		const Shader_xrLC& SH = F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)
			continue;
		b_material& M = lc_global_data()->materials()[F->dwMaterial];
		// Collect
		adjacent_vec.clear();
		for (int vit = 0; vit < 3; ++vit)
		{
			Vertex* V = F->v[vit];
			for (u32 adj = 0; adj < V->m_adjacents.size(); adj++)
			{
				adjacent_vec.push_back(V->m_adjacents[adj]);
			}
		}
		std::sort(adjacent_vec.begin(), adjacent_vec.end());
		adjacent_vec.erase(std::unique(adjacent_vec.begin(), adjacent_vec.end()), adjacent_vec.end());
		// Unique
		BOOL			bAlready = FALSE;
		for (u32 ait = 0; ait < adjacent_vec.size(); ++ait)
		{
			Face* Test = adjacent_vec[ait];
			if (Test == F) continue;
			if (!Test->flags.bProcessed) continue;
			if (FaceEqual(*F, *Test))
			{
				bAlready = TRUE; break;
			}
		}

		if (!bAlready)
		{
			u32 FaceIndex = counts_faces.load();
			if (!isCalculate)
			{
				F->flags.bProcessed = true;
				Fvector verts[3];
				verts[0] = F->v[0]->P; verts[1] = F->v[1]->P; verts[2] = F->v[2]->P;
				faces_embree[FaceIndex].SetVertexes_fast(verts, verts_embree, count_verts);
				(*dummy)[FaceIndex] = F;
			}
			counts_faces.fetch_add(1);
		}
	}

	auto& mu_refs = lc_global_data()->mu_refs();
	for (auto ref : mu_refs)
	{
		xr_vector<FaceDataIntel> temp_buffer;
		ref->export_cform_rcast_new(temp_buffer);

		for (auto F : temp_buffer)
		{
			u32 FaceIndex = counts_faces.load();

			if (!isCalculate)
			{
				Fvector verts[3];
				verts[0] = F.v1; verts[1] = F.v2; verts[2] = F.v3;
				(*dummy)[FaceIndex] = F.ptr;
				faces_embree[FaceIndex].SetVertexes_fast(verts, verts_embree, count_verts);
			}
			counts_faces.fetch_add(1);
		}
	}

}