#include "stdafx.h"
#include "../../xrEngine/xrlevel.h"

#include "xrThread.h"

#include "global_calculation_data.h"
#include "lightthread.h"
 
XRLC_LIGHT_API extern int	LIGHT_Count;

void	xrLightDO(u32 Samples)
{
	u32	range = gl_data.slots_data.size_z();
	LIGHT_Count = Samples;

	// Start threads, wait, continue --- perform all the work
	CThreadManager		Threads;
	CTimer				start_time;

	for (u32 thID = 0; thID < gCompilerMode.ThreadsNum; thID++)
	{
		CThread* T = xr_new<LightThread>( thID );
		T->thMessages = FALSE;
		T->thMonitor = FALSE;
		Threads.start(T, thID);
	}
	Threads.wait();

	Msg("%d seconds elapsed.", (start_time.GetElapsed_ms()) / 1000);
}
 



void xrCompileDO(u32 Samples)
{
	Phase		("Loading level...");
	gl_data.xrLoad	();

	Phase		("Lighting nodes...");
 	xrLightDO(Samples);

	gl_data.slots_data.Free();
	
}
