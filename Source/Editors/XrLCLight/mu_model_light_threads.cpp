#include	"stdafx.h"
#include	"mu_model_light_threads.h"


#include "xrface.h"
#include "xrMU_Model.h"
#include "xrMU_Model_Reference.h"

#include "xrlc_globaldata.h"

#include "mu_model_light.h"

#include "xrThread.h"
#include "../../xrcore/xrSyncronize.h"

 

CThreadManager			mu_base;
 
// mu-light
bool mu_models_local_calc_lightening = false;
xrCriticalSection		mu_models_local_calc_lightening_wait_lock;

void WaitMuModelsLocalCalcLightening()
{
	for(;;)
	{
		bool complited = false;
		Sleep(1000);
		mu_models_local_calc_lightening_wait_lock.Enter();
		complited = mu_models_local_calc_lightening;
		mu_models_local_calc_lightening_wait_lock.Leave();
		if(complited)
			break;
	}
}

void SetMuModelsLocalCalcLighteningCompleted()
{
	mu_models_local_calc_lightening_wait_lock.Enter();
	mu_models_local_calc_lightening = true;
	mu_models_local_calc_lightening_wait_lock.Leave();
}

/* OLD GSC
class CMULight	: public CThread
{
	u32			low;
	u32			high;
public:
	CMULight	(u32 ID, u32 _low, u32 _high) : CThread(ID)	{	thMessages	= FALSE; low=_low; high=_high;	}

	virtual void	Execute	()
	{
		// Priority
		SetThreadPriority	(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		Sleep				(0);

		// Light references
		for (u32 m=low; m<high; m++)
		{
		
			inlc_global_data()->mu_refs()[m]->calc_lighting	();
			thProgress							= (float(m-low)/float(high-low));
		}
	}
};
*/

#include <atomic>

std::atomic<int> task_id = 0;
 
//xr_vector<int>		task_pool_mu;

xrCriticalSection	task_CS;

//SE7KILLS
class CMULightBase : public CThread
{
public:
	CMULightBase(u32 ID) : CThread(ID) { thMessages = FALSE; }

	virtual void	Execute()
	{
		// Priority
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
 
		xrMU_Model* model = 0;

		for (;;)
		{
 			task_CS.Enter();
			int id = task_id.load();
 
			if (id < inlc_global_data()->mu_models().size())
				model = inlc_global_data()->mu_models()[id];
			else
				model = 0;

			task_id.fetch_add(1);
			task_CS.Leave();

			if (!model)
 				break;
 
			try
			{
				CTimer t;
				t.Start();
				model->calc_materials();
				model->calc_lighting();
				clMsg("Base mu-Model: %s, time: %d", model->m_name.c_str(), t.GetElapsed_ms());
			}
			catch (...)
			{
				clMsg("* ERROR: CMULight::Execute - calc_lighting");
			}

		}


	}
};

class CMULightRef : public CThread
{
public:
	CMULightRef(u32 ID) : CThread(ID) { thMessages = FALSE;}

	virtual void	Execute()
	{
		// Priority
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		//Sleep(0);

		xrMU_Reference* ref = 0;

		for (;;)
		{
  			task_CS.Enter();
			int id = task_id.load();
 
			if (id < inlc_global_data()->mu_refs().size())
				ref = inlc_global_data()->mu_refs()[id];
			else
				ref = 0;

			StatusNoMSG("IDS: %d/%d",id, inlc_global_data()->mu_refs().size());

			task_id.fetch_add(1);
			task_CS.Leave();

			if (!ref)
 				break;
 				  
			try
			{	
				CTimer t;
				t.Start();
				ref->calc_lighting();
				clMsg("MuRefModel: %s, time: %d", ref->model->m_name.c_str(), t.GetElapsed_ms());
			}
			catch (...)
			{
				clMsg("* ERROR: CMULight::Execute - calc_lighting");
			}
						
		}
		

	}
};

#include <execution>
#include "BuildArgs.h"

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

 
class CMUThread : public CThread
{
public:
	CMUThread	(u32 ID) : CThread(ID)
	{
		thMessages	= FALSE;
	}
	virtual void	Execute()
	{
		// Priority
		SetThreadPriority	(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		Sleep				(0);

		// Light models		
		task_id = 0;
	 
		Phase("LIGHT: Waiting for MU-First CALCMATERIALS threads...");
		/*
		CThreadManager thread_base;
		for (int TH = 0; TH < build_args->use_threads; TH++)
			thread_base.start(xr_new<CMULightBase>(TH), TH);

		thread_base.wait();
		 */
		for (const auto model : inlc_global_data()->mu_models())
		{
			model->calc_materials();
			model->calc_lighting();
		}

		SetMuModelsLocalCalcLighteningCompleted();

		// REFERENSE

		Phase("LIGHT: Waiting for MU-Secondary threads...");
 
		task_id = 0;
		
		CThreadManager			mu_secondary;

		for (int TH = 0; TH < build_args->use_threads; TH++)
			mu_secondary.start(xr_new<CMULightRef>(TH), TH);

		mu_secondary.wait(500);

	}
};


void	run_mu_base()
{
 	mu_base.start				(xr_new<CMUThread> (0), 0);
}

void	wait_mu_base_thread		()
{
	mu_base.wait				(500);
}
 