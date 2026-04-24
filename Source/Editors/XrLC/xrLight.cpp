#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "..\LauncherSDL\xrThread.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
#include "../xrLCLight/mu_model_light.h"
#include "../XrLCLight/embree_raytracing/EmbreeRayTrace.h"

#include "../XrLCLight/base_face.h"

#include <ppl.h>

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
 	const bool Cuda   = gCompilerMode.CUDA;
	const bool Embree = gCompilerMode.Embree;

	string128 tmp_phase;
	sprintf(tmp_phase, "LIGHT: LMaps (*%s*)", Cuda ? "CUDA" : Embree ? "Embree" : "Opcode");
	Phase(tmp_phase);

 	if (gCompilerMode.CUDA)
	{
		// Se7kills 
		CTimer start_time; start_time.Start();

		GPUTaskinSystem.RestartALL();
		GPUTaskinSystem.ColorsMapType = eDeflectors;
		GPUTaskinSystem.current_flags = (gCompilerMode.LC_NoSun ? LP_dont_sun : 0) | LP_UseFaceDisable;

		CTimer tStats; tStats.Start();
		auto ProcessDeflectors = [](xr_vector<CDeflector*>& deflectors)
		{
			std::atomic<u32> IndexTaskID = 0, IndexTaskApply = 0, IndexTaskExpand = 0;
			concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t TID)
			{
				while (true)
				{
					u32 Index = IndexTaskID.fetch_add(1);
					if (Index >= deflectors.size()) break;
					CDeflector* D = deflectors[Index];
 					D->LightGPU();

					AditionalData("*** [LMAPS] ID [%u/%u] W: %u | H: %u",
						Index, deflectors.size(), D->layer.width, D->layer.height);
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

					D->ApplyColors();
					D->ApplyExpandBordersGPU();

					AditionalData("*** [LMAPS] ApplyID [%u/%u] W: %u | H: %u",
						Index, deflectors.size(), D->layer.width, D->layer.height);
				}
			});
		};

		u32 AreaCollected = 0; u32 IndexD = 0;
		xr_vector<CDeflector*> deflectors_map;
		for (auto& D : lc_global_data()->g_deflectors())
		{
			// deflectors.
			if (AreaCollected > 8192 * 8192 * 20 || IndexD == lc_global_data()->g_deflectors().size())
			{
				// Lmaps Process
				ProcessDeflectors(deflectors_map);
				// Merge LMAPS
				xrPhase_MergeLM(deflectors_map);

				deflectors_map.clear();
				AreaCollected = 0;
			}

			IndexD++;
			AreaCollected += D->layer.Area();
			deflectors_map.push_back(D);
		}

		if (deflectors_map.size())
		{
			// Lmaps Process
			ProcessDeflectors(deflectors_map);
			// Merge LMAPS
			xrPhase_MergeLM(deflectors_map);

			deflectors_map.clear();
			AreaCollected = 0;
		}

		clMsg("%d lightmaps builded", lc_global_data()->lightmaps().size());
	}
	else
 	{
		// Main process (4 threads)
		Status("Lighting...");

		CTimer start_time; start_time.Start();
		ProcessLMAPS_CPU();
		clMsg("%f seconds", start_time.GetElapsed_sec());

		//****************************************** Merge LMAPS
		xrPhase_MergeLM(lc_global_data()->g_deflectors());
	}



	clMsg("Start Destroy Deflectors: Memory: %llu mb used", u32(GetHeapMemory() / 1024 / 1024));
	for (u32 it = 0; it < lc_global_data()->g_deflectors().size(); it++)
		xr_delete(lc_global_data()->g_deflectors()[it]);
	lc_global_data()->g_deflectors().clear();
	clMsg("End Destroy Deflectors: Memory: %llu mb used", u32(GetHeapMemory() / 1024 / 1024));
}

void CBuild::Light()
{	  
	//****************************************** Resolve materials
 	Phase("Resolving materials...");
 	xrPhase_ResolveMaterials();
	IsolateVertices(TRUE);

	//****************************************** UV mapping
 	Phase("Build UV mapping...");
 	xrPhase_UVmap();
	IsolateVertices(TRUE);

	//****************************************** Subdivide geometry
 	Phase("Subdividing geometry...");
 	xrPhase_Subdivide();

	//****************************************** Implicit
 
 	Phase("LIGHT: Implicit...");
 	ImplicitLighting();

	//****************************************** LMaps
  	LMaps();
	 
	//****************************************** MU-Models Processing
 	wait_mu_base();

	//****************************************** Vertex
 	Phase("LIGHT: Vertex...");
 	LightVertex();
 
 	Phase("Merging geometry...");
 	xrPhase_MergeGeometry();
 	  
	if (gCompilerMode.Embree)
		EmbreeMain.IntelEmbereUnloadAll();

	if (gCompilerMode.CUDA)
		GPUTaskinSystem.CleanupGPU();

	lc_global_data()->destroy_rcmodel();
}

void CBuild::LightVertex	()
{
	::LightVertex();
}