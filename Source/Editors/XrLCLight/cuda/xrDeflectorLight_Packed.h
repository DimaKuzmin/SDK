#pragma once
#include "../xrFace.h"
#include "../base_lighting.h"
#include "../base_color.h"

#include "../lm_layer.h"
#include "../uv_tri.h"
#include "../R_light.h"
#include "../xrMU_Model_Reference.h"

#include <ppl.h>
 
enum ColorsReturnType
{
	eImplicit,
	eDeflectors,
	eMumodel,
	eCommon
};

struct RayRecvestIndex
{
	void* Owner = 0;
	size_t  INDEX_TASK;

	// Task Pos, Dir, Skip
	Fvector P;
	Fvector N;
};

class PackedLighting
{
public:
	// Unordered for maps
	size_t MakeKey(u32 U, u32 V)
	{
		return (static_cast<u64>(U) << 32) | static_cast<u64>(V);
	}

	inline u32 GetU(u64 key)
	{
		return static_cast<u32>(key >> 32);
	}

	inline u32 GetV(u64 key)
	{
		return static_cast<u32>(key & 0xFFFFFFFFull);
	}

	void InitializeGPU();
	void CleanupGPU();

	/* —пециальные релизаци€ под разные типы освещени€ */
	ColorsReturnType ColorsMapType = eCommon;
	void LightPointPacked_add_task(size_t IndexTask, void* Refference, Fvector& P, Fvector& N, Face* skip);
	void LightPointPacked_run_tasks(bool need_clear = true);

	// Lightpoint Base
	xrCriticalSection										csEnter;
	concurrency::concurrent_unordered_map <size_t, base_color_c>	task_colors;

	// Reseting
	void RestartALL()
	{
		// start
		Recalculated = 0;
		current_flags = 0;

		// clearing pool
		task_colors.clear();
	}


	// Stats 
	bool	isInitializedGPU = false;
	u8	    current_flags = 0;

	// Stats
	u32		Recalculated = 0;
};

extern PackedLighting GPUTaskinSystem;


u32 GetFaceIndex(Face* F);
void SetFaceIndex(Face* F, u32 Index);