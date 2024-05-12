#include	"stdafx.h"
#include	"mu_model_light_threads.h"


#include "xrface.h"
#include "xrMU_Model.h"
#include "xrMU_Model_Reference.h"

#include "xrlc_globaldata.h"

#include "mu_model_light.h"
#include "mu_light_net.h"

//#include "mu_model_face.h"

#include "xrThread.h"
#include "../../xrcore/xrSyncronize.h"



CThreadManager			mu_base;

CThreadManager			mu_secondary;
CThreadManager			mu_calcmaterials;


 xrCriticalSection csMU;

int current_refthread;

class CMULightReferense	: public CThread
{
	u32			low;
	u32			high;
public:
	CMULightReferense(u32 ID) : CThread(ID)	{	thMessages	= FALSE; }

	virtual void	Execute	()
	{
		// Priority
		SetThreadPriority	(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		Sleep				(0);

		// Light references
		for (;;)
		{
			csMU.Enter();
			int ID = current_refthread;
			
			if (ID >= inlc_global_data()->mu_refs().size())
			{
				csMU.Leave();
				break;
			}

			//clMsg("Referense: %d / %d", ID, inlc_global_data()->mu_refs().size() );

			current_refthread++;
			csMU.Leave();
			
			float progress = float(ID / inlc_global_data()->mu_refs().size());
			thProgress = progress;
			
			// Progress(progress);

 			inlc_global_data()->mu_refs()[ID]->calc_lighting	();

			
			//thProgress							= (float(m-low)/float(high-low));
		}
	}
};


int current_thread_base;

class CMULightBase : public CThread
{
public:
	CMULightBase(u32 ID) : CThread(ID) { thMessages = FALSE; }

	virtual void	Execute()
	{
		// Priority
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		Sleep(0);

		// Light references
		for (;;)
		{
			csMU.Enter();
			int ID = current_thread_base;

			if (ID >= inlc_global_data()->mu_models().size())
			{
				csMU.Leave();
			
				break;
			}
			
			
		//	clMsg("Base: %d/%d", ID, inlc_global_data()->mu_models().size());
			current_thread_base++;
			csMU.Leave();
			float progress = float(ID / inlc_global_data()->mu_models().size());
			thProgress = progress;
			// Progress();

			inlc_global_data()->mu_models()[ID]->calc_materials();
			inlc_global_data()->mu_models()[ID]->calc_lighting();


			//thProgress							= (float(m-low)/float(high-low));
		}
	}
};

#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

extern int can_use_intel = 1;

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
		u32 threads = build_args->use_threads;
		can_use_intel = 0;
 
		Phase("LIGHT: Calculating Materials MU-thread...");
		
		if (!build_args->use_mt_calculation_materials)
		{
			int ID = 0;
 			for (auto model : inlc_global_data()->mu_models())
			{
				Progress(ID / inlc_global_data()->mu_models().size());
				model->calc_materials();
				model->calc_lighting();
				ID++;
			}
		}
		else
		{
			current_thread_base = 0;

			for (u32 thID = 0; thID < threads; thID++)
				mu_calcmaterials.start(xr_new<CMULightBase>(thID));

			mu_calcmaterials.wait(500);
		}

		can_use_intel = 1;

		// Light references
  	
		Phase("LIGHT: Calculating Referense MU-thread...");
  		current_refthread = 0;

		for (u32 thID=0; thID < threads; thID++)
			mu_secondary.start	( xr_new<CMULightReferense> (thID) );

		mu_secondary.wait(500);
 
	}
};


void	run_mu_base()
{
	
	mu_base.start				(xr_new<CMUThread> (0));
}

void	wait_mu_base_thread		()
{
	mu_base.wait				(500);
}

 