#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrThread.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
#include "net.h"
//#include "../xrLCLight/net_task_manager.h"
#include "../xrLCLight/lcnet_task_manager.h"
#include "../xrLCLight/mu_model_light.h"

#include "../XrLCLight/net_cl_data_prepare.h"


#include "../XrLCLight/base_face.h"



xrCriticalSection	task_CS
#ifdef PROFILE_CRITICAL_SECTIONS
	(MUTEX_PROFILE_ID(task_C_S))
#endif // PROFILE_CRITICAL_SECTIONS
;

xr_vector<int>		task_pool;
 
XRLC_LIGHT_API void InitDB(CDB::COLLIDER* DB, bool print);

class CLMThread		: public CThread
{
private:
	HASH			H;
	CDB::COLLIDER	DB;
	base_lighting	LightsSelected;

public:
	CLMThread	(u32 ID) : CThread(ID)
	{
		// thMonitor= TRUE;
		thMessages	= FALSE;
	}

	virtual void	Execute()
	{
		InitDB(&DB);

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

			/*
			if (task_pool.size() % 1 == 0)
				StatusNoMSG("DEFL[%d]/[%d], layer w[%d], h[%d]", 
					lc_global_data()->g_deflectors().size() - task_pool.size(), 
					lc_global_data()->g_deflectors().size(),
					D->layer.width, D->layer.height
				);
			*/
			
			task_pool.pop_back	();
			task_CS.Leave		();

			//base_Vertex* vert = (base_Vertex*) D->UVpolys.front().owner;
			//if (vert)
			//Msg("POS [%.2f][%.2f][%.2f]", vert->P.x, vert->P.y, vert->P.z );

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
		InitDB(&DB);

		CDeflector* D = 0;

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

			/*
			if (task_pool.size() % 1 == 0)
			StatusNoMSG("DEFL[%d]/[%d], layer w[%d], h[%d]",
				lc_global_data()->g_deflectors().size() - task_pool.size(),
				lc_global_data()->g_deflectors().size(),
				D->layer.width, D->layer.height
			);
			*/

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


int THREADS_COUNT()
{
	LPCSTR str = strstr(Core.Params, "-th");

	if (str)
	{
		int count = 0;
		LPCSTR new_str = str + 3;
		sscanf(new_str, "%d", &count);
		return count;
	}

	return 4;
}

#define TH_NUM THREADS_COUNT()
#include "..\XrLCLight\xrHardwareLight.h"


void IntelEmbereUNLOAD();
void IntelClearTimers(LPCSTR name);

void IntelEmbereLOAD();
XRLC_LIGHT_API extern bool use_intel;

#include <tbb/parallel_for_each.h>
#include <random>

  
void	CBuild::LMapsLocal				()
{
		FPU::m64r		();
		
		mem_Compact		();
 
		// Randomize deflectors
#ifndef NET_CMP
		 
		std::sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), [] (const CDeflector* defl, const CDeflector* defl2) 
		{
			return defl->similar_pos(*defl2, 0.1f);	    
		});
		 
		for (auto defl : lc_global_data()->g_deflectors())
		{
			Msg_IN_FILE("Defl Sphere: [%f][%f][%f]", VPUSH(defl->Sphere.P) );
		}

		//std::shuffle (lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), std::random_device() );
#endif

 
		 
#ifndef NET_CMP	
for(u32 dit = 0; dit<lc_global_data()->g_deflectors().size(); dit++)	
		task_pool.push_back(dit);

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
			
		int th = TH_NUM;
		 
		for (int L = 0; L < th; L++)
			threads.start(xr_new<CLMThread>(L));
		threads.wait(500);
		
#ifndef OLD_METHOD_GPU_COMPUTE		 
		if (xrHardwareLight::IsEnabled())
			GPU_Calculation();

		for (u32 dit = 0; dit < lc_global_data()->g_deflectors().size(); dit++)
			task_pool.push_back(dit);

		for (int L = 0; L < th; L++)
			threads.start(xr_new<CLMThreadEndJob>(L));
	
		threads.wait(500);
#endif

		clMsg			("%f seconds",start_time.GetElapsed_sec());


}

void	CBuild::LMaps					()
{
		//****************************************** Lmaps

	//DeflectorsStats ();
#ifndef NET_CMP
	if(g_build_options.b_net_light)

		//net_light ();
		lc_net::net_lightmaps ();
	else{
		LMapsLocal();
	}
#else
	create_net_task_manager();
	get_net_task_manager()->create_global_data_write(pBuild->path);
	LMapsLocal();
	get_net_task_manager()->run();
	destroy_net_task_manager();
	//net_light ();
#endif

}
void XRLC_LIGHT_API ImplicitNetWait();

void CBuild::Light()
{
	Msg("QUALYTI: %d, pixel: %d, jitter: %d", g_params().m_quality, g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);

	if (g_params().m_quality != ebqDraft && !strstr(Core.Params, "-no_light"))
	{
		IntelClearTimers("Pre Implicit");

		//****************************************** Implicit
		{
			FPU::m64r();
			string128 tmp; sprintf(tmp, "LIGHT: Implicit...[%s]", use_intel ? "intel" : "opcode");
			Phase(tmp);
			mem_Compact();
			ImplicitLighting();
		}

		IntelClearTimers("Implicit");

 		{
 			string128 tmp; sprintf(tmp, "LIGHT: LMaps...[%s]", use_intel ? "intel" : "opcode");
			Phase			(tmp);
			LMaps();

			//****************************************** Vertex
			FPU::m64r();
			Phase("LIGHT: Vertex...");
			mem_Compact();

			IntelClearTimers("LMAPS");
			LightVertex();


			ImplicitNetWait();
			lc_net::get_task_manager().wait_all();
			lc_net::get_task_manager().release();

			//****************************************** Merge LMAPS
			{
				FPU::m64r();
				Phase("LIGHT: Merging lightmaps...");
				mem_Compact();

				xrPhase_MergeLM();
			}

		}

		IntelClearTimers("LM Vertex");
	}

	//****************************************** Starting MU
	if ( !strstr(Core.Params, "-no_light_mu"))
	{
		FPU::m64r();
		Phase("LIGHT: Starting MU...");
		mem_Compact();
		Light_prepare();
		if (g_build_options.b_net_light)
		{
			lc_global_data()->mu_models_calc_materials();
			RunNetCompileDataPrepare();
		}
		StartMu();


		//****************************************** Wait for MU
		FPU::m64r();
				
 		string128 tmp; sprintf(tmp, "LIGHT: Waiting MU...[%s]", use_intel ? "intel" : "opcode");
		Phase(tmp);
		mem_Compact();
		wait_mu_base();

		if (!g_build_options.b_net_light)
		{
			Phase("LIGHT: Waiting for MU-Secondary threads...");
			wait_mu_secondary();
		}

	}

	IntelClearTimers("LM MuModels");

	if (use_intel)
		IntelEmbereUNLOAD();
}

void CBuild::LightVertex	()
{
	::LightVertex(!!g_build_options.b_net_light);
}