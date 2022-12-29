#pragma once

#ifndef _MAYA_EXPORT

	#ifdef XRETOOLS_EXPORTS
		#define ETOOLS_API __declspec( dllexport )
	#else
		#define ETOOLS_API __declspec( dllimport )
	#endif
#else
	#define ETOOLS_API
#endif

#include "ArbitraryList.h"

#pragma pack(push,1)
struct VIPM_SWR
{
	u32		offset;					// Offset of the first index in the index buffer to start at (note! no retrictions. Can easily be >64k)
	u16		num_tris;				// Number of tris to render (most cards can't do more than 65536)
	u16		num_verts;				// Number of vertices to render with (using WORD indices)
};
#pragma pack(pop)

struct VIPM_Result
{
	ArbitraryList<u16>		permute_verts;
	ArbitraryList<VIPM_SWR>	swr_records;// The records of the collapses.
	ArbitraryList<u16>		indices;
	~VIPM_Result()
	{
		permute_verts.resize(0);
		swr_records.resize	(0);
		indices.resize		(0);
	}
};

extern "C" 
{
	ETOOLS_API void			  VIPM_Init			();
	ETOOLS_API void			  VIPM_AppendVertex	(const Fvector3& pt, const Fvector2& uv);
	ETOOLS_API void			  VIPM_AppendFace		(u16 v0, u16 v1, u16 v2);
	ETOOLS_API VIPM_Result*	  VIPM_Convert			(u32 max_sliding_window=u32(-1), float error_tolerance=0.1f, u32 optimize_vertex_order=1);
	ETOOLS_API void			  VIPM_Destroy			();
};

class Object;
class MeshPt;

class ETOOLS_API VIPM_MultiTH
{
public:
	Object* g_mt_pObject = 0;
	ArbitraryList<MeshPt*>	g_mt_ppTempPts = 0;
	VIPM_Result* g_mt_pResult = 0;

	void			  VIPM_Init	(int th);
	void			  VIPM_AppendVertex(const Fvector3& pt, const Fvector2& uv);
	void			  VIPM_AppendFace(u16 v0, u16 v1, u16 v2);
	VIPM_Result*	  VIPM_Convert(u32 max_sliding_window = u32(-1), float error_tolerance = 0.1f, u32 optimize_vertex_order = 1);
	void			  VIPM_Destroy();

	ICF void CalcALL(Object* m_pObject, u32 max_sliding_window = u32(-1), float m_fSlidingWindowErrorTolerance = 1.f);
};
 