#include "stdafx.h"
#include "mu_model_light.h"

#include "xrFace.h"
#include "xrMU_Model.h"
#include "xrMU_Model_Reference.h"
#include "xrlc_globaldata.h"
#include "cuda/xrDeflectorLight_Packed.h"
#include "light_point.h"

// mu-light
#include <atomic>
#include <ppl.h>
std::atomic<u32> task_id = 0;

void	wait_mu_base		()
{
	// Light models		
	Phase("LIGHT: Waiting for MU-First");
	task_id = 0;

	// Basic Types
	concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [](size_t THID)
	{
		xrMU_Model* model = 0;
		while (true)
		{
			int id = task_id.fetch_add(1);
			if (id >= inlc_global_data()->mu_models().size()) break;

			model = inlc_global_data()->mu_models()[id];
			model->calc_materials();
			model->calc_lighting();
		}
	});


	Phase("LIGHT: Waiting for MU-Refs...");

	// Refference Calculation
	if (gCompilerMode.CUDA)
	{
		GPUTaskinSystem.RestartALL();
		GPUTaskinSystem.ColorsMapType = eMumodel;
		GPUTaskinSystem.current_flags = (gCompilerMode.LC_NoSun ? LP_dont_sun : 0) | LP_DEFAULT;

		// Gathering
		CTimer tStats; tStats.Start();

		std::atomic<u32> REF_INDEX = 0;
		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t ThreadID)
			{
				while (true)
				{
					u32 IndexTask = REF_INDEX.fetch_add(1);
					if (IndexTask >= inlc_global_data()->mu_refs().size()) break;

					AditionalData("REF LIGHT: %u/%u", IndexTask, inlc_global_data()->mu_refs().size());
					auto MRef = inlc_global_data()->mu_refs()[IndexTask];
					MRef->calc_lighting_cuda_1();
				};

				// Завершаем накопленые данные
				GPUTaskinSystem.LightPointPacked_run_tasks();
			});
		Msg("[MURefs] Elapsed For Compute: %u ms", tStats.GetElapsed_ms());

		// APPLY

		tStats.Start();
		REF_INDEX = 0;
		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t ThreadID)
			{
				while (true)
				{
					u32 Index = REF_INDEX.fetch_add(1);
					if (Index >= inlc_global_data()->mu_refs().size()) break;

					auto REF = inlc_global_data()->mu_refs()[Index];

					REF->calc_lighting_cuda_2();
					REF->calc_lighting_cuda_3();

					AditionalData("REF LIGHT APPLY: %u/%u", Index, inlc_global_data()->mu_refs().size());
				}
			});

		Msg("[MURefs] Elapsed For Apply Colors: %u ms", tStats.GetElapsed_ms());

		GPUTaskinSystem.RestartALL(); // Выгружаем все Это последнее освещение 
	}
	else
	{
 		// REFERENSE
 		task_id = 0;
		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [](size_t THID)
			{
				// Priority
				xrMU_Reference* ref = 0;
				while (true)
				{
					int id = task_id.fetch_add(1);
					if (id >= inlc_global_data()->mu_refs().size()) break;
					AditionalData("IDS: %d/%d", id, inlc_global_data()->mu_refs().size());

					ref = inlc_global_data()->mu_refs()[id];
					ref->calc_lighting();
				}
			});
	}

}