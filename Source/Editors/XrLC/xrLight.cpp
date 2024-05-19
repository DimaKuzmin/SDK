#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrThread.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
 
//#include "../xrLCLight/net_task_manager.h"
 #include "../xrLCLight/mu_model_light.h"

 

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

			 
 			StatusNoMSG("DEFL[%d]/[%d], layer w[%d], h[%d]", 
				lc_global_data()->g_deflectors().size() - task_pool.size(), 
				lc_global_data()->g_deflectors().size(),
				D->layer.width, D->layer.height
			);
			 
			
			task_pool.pop_back	();
			task_CS.Leave		();
 
			// Perform operation
			try 
			{
				D->Light	(thID, &DB,&LightsSelected,H);
			} 
			catch (...)
			{
				clMsg("* ERROR: CLMThread::Execute - light");
			}
		}
	}
};

class CLMThreadEndJob : public CThread
{
private:
	HASH			H;
	CDB::COLLIDER	DB;
	base_lighting	LightsSelected;

public:
	CLMThreadEndJob(u32 ID) : CThread(ID)
	{
		// thMonitor= TRUE;
		thMessages = FALSE;
	}

	virtual void	Execute()
	{
 
		CDeflector* D = 0;
		Status("Thread Execute: %d", thID);
		for (;;)
		{
			// Get task
			task_CS.Enter();

			thProgress = 1.f - float(task_pool.size()) / float(lc_global_data()->g_deflectors().size());

			if (task_pool.empty())
			{
				task_CS.Leave();
				return;
			}

			D = lc_global_data()->g_deflectors()[task_pool.back()];

			 
		//	if (task_pool.size() % 1 == 0)
			Status("DEFL[%d]/[%d], layer w[%d], h[%d]",
				lc_global_data()->g_deflectors().size() - task_pool.size(),
				lc_global_data()->g_deflectors().size(),
				D->layer.width, D->layer.height
			);
		 

			task_pool.pop_back();
			task_CS.Leave();


			//base_Vertex* vert = (base_Vertex*) D->UVpolys.front().owner;
			//if (vert)
			//Msg("POS [%.2f][%.2f][%.2f]", vert->P.x, vert->P.y, vert->P.z );
			
			// Perform operation
			try
			{
				D->LightEnd(thID, &DB, &LightsSelected, H);
			}
			catch (...)
			{
				clMsg("* ERROR: CLMThread::Execute - light");
			}
		}
	}
};


#ifndef DevCPU
	#include "..\XrLCLight\xrHardwareLight.h"
#endif


void IntelEmbereUNLOAD();


#include <tbb/parallel_for_each.h>
#include <random>

  
void	CBuild::LMapsLocal				()
{
		FPU::m64r		();
		
		mem_Compact		();
 
		// Randomize deflectors
#ifndef NET_CMP
		 
		Status("Sorting Deflectors : Start");
		std::sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), [] (const CDeflector* defl, const CDeflector* defl2) 
		{
			return defl->similar_pos(*defl2, 0.1f);	    
		});
		Status("Sorting Deflectors : End");

 #endif

 
		 
#ifndef NET_CMP	
		Status(" Deflectors Move TO POOL: Start");
for(u32 dit = 0; dit<lc_global_data()->g_deflectors().size(); dit++)	
		task_pool.push_back(dit);
		Status(" Deflectors Move TO POOL: End");


//for (u32 dit = lc_global_data()->g_deflectors().size(); dit > 0; dit--)
//	 task_pool.push_back(dit);
#else
		task_pool.push_back(14);
		task_pool.push_back(16);
#endif
 
		// Main process (4 threads) (-th MAX_THREADS)
		Status			("Lighting...");
		CThreadManager	threads;
 		CTimer	start_time;	
		start_time.Start();				
			
		int th = build_args->use_threads;
		 
		for (int L = 0; L < th; L++)
			threads.start(xr_new<CLMThread>(L), L);
		threads.wait(500);
		
#ifndef DevCPU
#ifndef OLD_METHOD_GPU_COMPUTE		 
		if (xrHardwareLight::IsEnabled())
		{
			GPU_Calculation();

			for (u32 dit = 0; dit < lc_global_data()->g_deflectors().size(); dit++)
				task_pool.push_back(dit);

			for (int L = 0; L < th; L++)
				threads.start(xr_new<CLMThreadEndJob>(L));
			threads.wait(500);
		}
	
#endif
#endif

		clMsg			("%f seconds",start_time.GetElapsed_sec());


}

void	CBuild::LMaps					()
{
	//****************************************** Lmaps
	LMapsLocal();
}
 
void CBuild::Light()
{
	Msg("QUALYTI: %d, pixel: %d, jitter: %d", g_params().m_quality, g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);

	if (g_params().m_quality != ebqDraft )	 
	{
	

 
		//****************************************** Implicit
	
		if (!build_args->off_impl)
		{
			FPU::m64r();
			string128 tmp; sprintf(tmp, "LIGHT: Implicit...[%s]",  build_args->use_IMPLICIT_Stage ? "intel" : "opcode");
			Phase(tmp);
			mem_Compact();
			ImplicitLighting();
		}

		if (!build_args->off_lmaps)
 		{
 			string128 tmp; sprintf(tmp, "LIGHT: LMaps...[%s]", build_args->use_LMAPS_Stage ? "intel" : "opcode");
			Phase			(tmp);
			LMaps();

			//****************************************** Vertex
			FPU::m64r();
			Phase("LIGHT: Vertex...");
			mem_Compact();

 			LightVertex();
 
			//****************************************** Merge LMAPS
			{
				FPU::m64r();
				Phase("LIGHT: Merging lightmaps...");
				mem_Compact();

				xrPhase_MergeLM();
			}

		}

 	}

	//****************************************** Starting MU
	if (!build_args->off_mulitght)
	{
		FPU::m64r();
		Phase("LIGHT: Starting MU...");
		mem_Compact();
		Light_prepare();
		//****************************************** Wait for MU
		FPU::m64r();

		string128 tmp; sprintf(tmp, "LIGHT: Waiting MU...[%s]", build_args->use_embree && build_args->use_MU_Lighting ? "intel" : "opcode");
		Phase(tmp);

		wait_mu_base();
	}

 
	if (build_args->use_embree)
		IntelEmbereUNLOAD();
}

void CBuild::LightVertex	()
{
	::LightVertex();
}