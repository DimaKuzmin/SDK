#pragma once

#include "R_light.h"
#include "base_lighting.h"
#include "base_color.h"
#include "../XrCDB/xrCDB.h"


#include "embree4/rtcore.h"
#pragma comment(lib, "embree4.lib")
#include "xrDeflector.h"

#include <atomic>
// Vertex, Tri Buffers
namespace Embree
{

	struct VertexEmbree
	{
		float x, y, z;

		void Set(Fvector& vertex);
 		Fvector Get();
	};

	struct TriEmbree
	{
		uint32_t point1, point2, point3;
		void SetVertexes(CDB::TRI& triangle, Fvector* verts, VertexEmbree* emb_verts, size_t& last_index);
		void SetVertexes_new(FaceDataIntel& data, VertexEmbree* emb_verts, size_t& last_index);

		void SetVertexes_fast(Fvector* Vs, VertexEmbree* emb_verts, std::atomic<size_t>& last_index);
	};


	// ВАЖНЫЙ ПАРАМЕТР TNEAR Для пересечения с водой
	void SetRay1(RTCRay& rayhit, Fvector& pos, Fvector& dir, float near_, float range);
 	void SetRay1(RTCRayHit& rayhit, Fvector& pos, Fvector& dir, float near_, float range);


	void errorFunction(void* userPtr, enum RTCError error, const char* str);
	void IntelEmbreeSettings(RTCDevice& device, bool avx, bool sse);

	void GetGlobalData(bool isTransp, bool isCalculate, std::atomic<size_t>& counts_faces, Embree::VertexEmbree* verts_embree, Embree::TriEmbree* faces_embree, xr_vector<void*>* dummy);
}
 
extern XRLC_LIGHT_API float RaytraceEmbreeProcess(R_Light& L, Fvector& P, Fvector& N, float range, void* skip);
  