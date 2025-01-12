#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrThread.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
 
//#include "../xrLCLight/net_task_manager.h"
#include "../xrLCLight/mu_model_light.h"
#include "../XrLCLight/xrLight_Embree.h"

#include "../XrLCLight/base_face.h"

#include "../XrLCLight/BuildArgs.h"

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

xrCriticalSection	task_CS
#ifdef PROFILE_CRITICAL_SECTIONS
	(MUTEX_PROFILE_ID(task_C_S))
#endif // PROFILE_CRITICAL_SECTIONS
;

xr_vector<int>		task_pool;
 
 
class CLMThread		: public CThread
{
private:
	HASH			H;
	CDB::COLLIDER	DB;
	base_lighting	LightsSelected;

public:
	CLMThread	(u32 ID) : CThread(ID)
	{
 		thMessages	= TRUE;
	}

	virtual void	Execute()
	{
 
		CDeflector* D	= 0;
 
		for (;;) 
		{
			// Get task
			task_CS.Enter		();
			
			thProgress			= 1.f - float(task_pool.size())/float(lc_global_data()->g_deflectors().size());
		
			if (task_pool.empty())	
			{
				task_CS.Leave		();
				return;
			}

			D					= lc_global_data()->g_deflectors()[task_pool.back()];

			 
 			// StatusNoMSG("DEFL[%d]/[%d], layer w[%d], h[%d]", 
			// 	lc_global_data()->g_deflectors().size() - task_pool.size(), 
			// 	lc_global_data()->g_deflectors().size(),
			// 	D->layer.width, D->layer.height
			// );
			 
			
			task_pool.pop_back	();
			task_CS.Leave		();
 
			// Perform operation
			try 
			{
				D->Light	(&DB,&LightsSelected,H);
			} 
			catch (...)
			{
				clMsg("* ERROR: CLMThread::Execute - light");
			}
		}
	}
};

void IntelEmbereUNLOAD();
 
#include "ppl.h"
#include <atomic>
  
void	CBuild::LMapsLocal				()
{
	FPU::m64r		();
		
	mem_Compact		();

 	std::sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), [](const CDeflector* defl, const CDeflector* defl2)
	{
		return defl->similar_pos(*defl2, 0.1f);
	});
	 
	CTimer	start_time;
	start_time.Start();

	// Main process (4 threads) (-th MAX_THREADS)
	Status("Lighting...");
 
	/// CThreadManager	threads;
	/// 	for (u32 dit = 0; dit < lc_global_data()->g_deflectors().size(); dit++)
	/// task_pool.push_back(dit);
	/// int th = build_args->use_threads;
	/// 
	/// for (int L = 0; L < th; L++)
	/// 	threads.start(xr_new<CLMThread>(L), L);
	/// threads.wait(500);
	 

	thread_local HASH			H;
	thread_local CDB::COLLIDER	DB;
	thread_local base_lighting	LightsSelected;

	u32 Progress = 0;
	std::atomic<int> processed;

	u32 LastProgressBar = 0;

	u32 MaxSize = lc_global_data()->g_deflectors().size();
	   
	concurrency::parallel_for(size_t(0), size_t(lc_global_data()->g_deflectors().size()), [&](size_t ID)
	{
 		
		// Get task
 		CDeflector* D = lc_global_data()->g_deflectors()[ID];
		  
		// Perform operation
		try
		{
			D->Light(&DB, &LightsSelected, H);
		}
		catch (...)
		{
			clMsg("* ERROR: CLMThread::Execute - light");
		}
	
 		processed.fetch_add(1);
		
		if (LastProgressBar < processed)
		{
			StatusNoMSG("Deflectors Ended : %u ", processed.load());
			LastProgressBar = processed + 4096;
		}
		
		// StatusNoMSG("DEFL[%d]/[%d]", ID, lc_global_data()->g_deflectors().size());
	});



	clMsg("%f seconds", start_time.GetElapsed_sec());
}

void	CBuild::LMaps					()
{
	//****************************************** Lmaps
	clMsg("Start Processing LMAPS: ");
	LMapsLocal();
}
 
extern void log_vminfo_new(LPCSTR msg);

void CBuild::RunMuModels()
{

	//****************************************** Starting MU
 	{
		FPU::m64r();
		Phase("LIGHT: Starting MU...");
		mem_Compact();
		Light_prepare();
		//****************************************** Wait for MU
		FPU::m64r();

		string128 tmp; sprintf(tmp, "LIGHT: Waiting MU...[%s]", build_args->use_embree ? "intel" : "opcode");
		Phase(tmp);

		wait_mu_base();
	}
}


#include "xrLC.h"
 
void CBuild::Light()
{
	Msg("QUALYTI: %d, pixel: %d, jitter: %d", g_params().m_quality, g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);

	if (g_params().m_quality != ebqDraft)
	{

		if (build_args->run_mu_first)
		{
			RunMuModels();
			log_vminfo_new("MU-MODELS Memory");
		}
		//****************************************** Implicit

 		{
			FPU::m64r();
			string128 tmp; sprintf(tmp, "LIGHT: Implicit...[%s]", build_args->use_embree ? "intel" : "opcode");
			Phase(tmp);
			mem_Compact();
			ImplicitLighting();
			log_vminfo_new("Implicit Memory");
		}

 		{
 
			string128 tmp; sprintf(tmp, "LIGHT: LMaps...[%s]", build_args->use_embree ? "intel" : "opcode");
			Phase(tmp);
			LMaps();
			log_vminfo_new("LMAPS Memory");


			//****************************************** Vertex
			FPU::m64r();
			Phase("LIGHT: Vertex...");
			mem_Compact();
			LightVertex();
			log_vminfo_new("Vertex Light Memory");

			//****************************************** Merge LMAPS
			{
				FPU::m64r();
				Phase("LIGHT: Merging lightmaps...");
				mem_Compact();
				xrPhase_MergeLM();
				log_vminfo_new("Merge LIGHTMAPS Memory");
			}

		}

		if (!build_args->run_mu_first)
		{
 			RunMuModels();
			log_vminfo_new("MU-MODELS Memory");
		}
 	}
  
	if (build_args->use_embree)
		IntelEmbereUNLOAD();
}

void CBuild::LightVertex	()
{
	::LightVertex();
}