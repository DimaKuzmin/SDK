#include "stdafx.h"
#include "../../xrEngine/xrlevel.h"

#include "xrThread.h"

#include "global_calculation_data.h"
#include "lightthread.h"
#include "xrLightDoNet.h"

//#define STD_THREAD 

#define NUM_THREADS		8


#ifndef STD_THREAD
void	xrLightDO()
{
	u32	range = gl_data.slots_data.size_z();

	// Start threads, wait, continue --- perform all the work
	CThreadManager		Threads;
	CTimer				start_time;
	u32	stride = range / NUM_THREADS;
	u32	last = range - stride * (NUM_THREADS - 1);

	for (u32 thID = 0; thID < NUM_THREADS; thID++)
	{
		CThread* T = xr_new<LightThread>(thID, thID * stride, thID * stride + ((thID == (NUM_THREADS - 1)) ? last : stride));
		T->thMessages = FALSE;
		T->thMonitor = FALSE;
		Threads.start(T);
	}
	Threads.wait();
	Msg("%d seconds elapsed.", (start_time.GetElapsed_ms()) / 1000);
}
#else
xr_vector<u32> thread_work;
xrCriticalSection csDO;

void MT()
{
	DWORDVec	box_result;

	CDB::COLLIDER		DB;
	DB.ray_options(CDB::OPT_CULL);
	DB.box_options(CDB::OPT_FULL_TEST);
	base_lighting		Selected;

	for (;;)
	{
		csDO.Enter();
		if (thread_work.empty())
		{
			csDO.Leave();
			break;
		}

		int z = thread_work.back();
		thread_work.pop_back();

		csDO.Leave();

		//	Status("Work: %d", z);


		for (int x = 0; x < gl_data.slots_data.size_x(); x++)
		{
			DetailSlot& DS = gl_data.slots_data.get_slot(x, z);

			if (!detail_slot_process(x, z, DS))
				continue;

			if (!detail_slot_calculate(x, z, DS, box_result, DB, Selected))
				continue; //?

			gl_data.slots_data.set_slot_calculated(x, z);
		}
	}
}

#include <thread>

void xrLightDO()
{
	for (u32 _z = gl_data.slots_data.size_z() - 1; _z > 0; _z--)
	{
		thread_work.push_back(_z);
	}

	std::thread* th = new std::thread[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; i++)
		th[i] = std::thread(MT);
	for (int i = 0; i < NUM_THREADS; i++)
		th[i].join();
}
#endif // !STD_THREAD
 



void xrCompileDO( bool net )
{
	Phase		("Loading level...");
	gl_data.xrLoad	();

	Phase		("Lighting nodes...");
	if( net )
		lc_net::xrNetDOLight();
	else
		xrLightDO();

	gl_data.slots_data.Free();
	
}
