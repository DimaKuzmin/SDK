#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
#include "../xrLCLight/mu_model_light.h"
#include "../XrLCLight/embree_raytracing/EmbreeRayTrace.h"

#include "../XrLCLight/base_face.h"

#include <ppl.h>
extern void ImplicitLightingExec();

void CBuild::ProcessLMAPS_CPU()
{
	thread_local CDB::COLLIDER	DB;
	thread_local base_lighting	LightsSelected;

	std::atomic<u32> CurrentIndex = 0;
	concurrency::parallel_for(0, gCompilerMode.ThreadsNum, [&](int ThreadID)
		{
			while (true)
			{
				// Get task
				u32 IndexTask = CurrentIndex.fetch_add(1); //-> prev ID
 				if (IndexTask >= lc_global_data()->g_deflectors().size()) break;
				
				Progress(float(IndexTask) / float(lc_global_data()->g_deflectors().size()));
				AditionalData("Deflectors: %u / %u", IndexTask, lc_global_data()->g_deflectors().size());

				CDeflector* D = lc_global_data()->g_deflectors()[IndexTask];
 				D->Light(&DB, &LightsSelected);
			}
		}
	);
};

#include "../xrLCLight/cuda/xrDeflectorLight_Packed.h"
#include "../xrLCLight/light_point.h"

void	CBuild::LMaps()
{
	Status("Lighting...");

	// Sorting Deflectors for Saving !
 	if (gCompilerMode.CUDA)
	{
		// Se7kills 
		CTimer start_time; start_time.Start();

		GPUTaskinSystem.RestartALL();
		GPUTaskinSystem.ColorsMapType = eDeflectors;
		GPUTaskinSystem.current_flags = (gCompilerMode.LC_NoSun ? LP_dont_sun : 0) | LP_UseFaceDisable;

		CTimer tStats; tStats.Start();

		auto& deflectors = lc_global_data()->g_deflectors();
		std::atomic<u32> IndexTaskID = 0, IndexTaskApply = 0, IndexTaskExpand = 0;
		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t TID)
			{
				while (true)
				{
					u32 Index = IndexTaskID.fetch_add(1);
					if (Index >= deflectors.size()) break;
					CDeflector* D = deflectors[Index];
					if (D->bLightProcessed) continue;	// Временно в буфере находится !

					D->LightGPU();
					AditionalData("*** [LMAPS] ID [%u/%u]", Index, deflectors.size());
				}

				// Система тасков щас иная
				GPUTaskinSystem.LightPointPacked_run_tasks();
			});

		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t TID)
			{
				while (true)
				{
					u32 Index = IndexTaskApply.fetch_add(1);
					if (Index >= deflectors.size()) break;
					CDeflector* D = deflectors[Index];
					if (D->bLightProcessed) continue;	// Временно в буфере находится !

					D->ApplyColors();
					D->ApplyExpandBordersGPU();

					AditionalData("*** [LMAPS] ApplyID [%u/%u]", Index, deflectors.size());
				}
			});
	}
	else
 	{
		// Main process (4 threads)
		ProcessLMAPS_CPU();
	}

	// Закрыть и записать !
 	xrPhase_MergeLM(lc_global_data()->g_deflectors());

	clMsg("Start Destroy Deflectors: Memory: %llu mb used", u32(GetHeapMemory() / 1024 / 1024));
	for (u32 it = 0; it < lc_global_data()->g_deflectors().size(); it++)
		xr_delete(lc_global_data()->g_deflectors()[it]);
	lc_global_data()->g_deflectors().clear();
	clMsg("End Destroy Deflectors: Memory: %llu mb used", u32(GetHeapMemory() / 1024 / 1024));
}


void CBuild::Light()
{
	auto BuildRayTraceModel = [this]()
	{
		if (gCompilerMode.CUDA || gCompilerMode.Embree)
			InitializeEmbreeDevice();

		if (gCompilerMode.CUDA)
			GPUTaskinSystem.InitializeGPU();
		else if (gCompilerMode.Embree)
			EmbreeMain.InitializeGeometry();
	};

	auto BuildingUV = [this]() 
	{
 		Phase("Building - UV ...");
		xrPhase_ResolveMaterials();
		IsolateVertices(TRUE);

		xrPhase_UVmap();
		IsolateVertices(TRUE);

		xrPhase_Subdivide();
	};
	  
	Phase("Building normals...");
	CalcNormals();

	Light_prepare();				// Помечаем треугольники bOpacue !


 	// ***************************************** Raytrace Model
	BuildRayTraceModel();

	//****************************************** UV mapping
	BuildingUV();

	Phase("Adaptive HT...");
	xrPhase_AdaptiveHT_calculate();

 	//****************************************** Implicit
 	Phase("LIGHT: Implicit...");
	ImplicitLightingExec();

	//****************************************** LMaps
	Phase("LIGHT: Lmaps...");
	LMaps();
	 
	//****************************************** Vertex
	Phase("LIGHT: Vertex...");
	::LightVertex();

	//****************************************** MU-Models Processing
	wait_mu_base();
  
 	Phase("Merging geometry...");
 	xrPhase_MergeGeometry();
 	  
 	EmbreeMain.IntelEmbereUnloadData();
  	GPUTaskinSystem.CleanupGPU();
}
 