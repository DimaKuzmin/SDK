#include "stdafx.h"
#include "xrThread.h"
#include <mutex>
#include <atomic>

void	CThread::startup(void* P)
{
	CThread* T = (CThread*)P;

	if (T->thMessages)
		clMsg("* THREAD #%d: Started.", T->thID);
	FPU::m64r();

	DWORD_PTR affinityMask = 1ull << T->thID;
	HANDLE threadHandle = GetCurrentThread();
	SetThreadAffinityMask(threadHandle, affinityMask);

	T->Execute();
	T->thCompleted = TRUE;
	if (T->thMessages)
		clMsg("* THREAD #%d: Task Completed.", T->thID);
}
 
void	CThreadManager::start	(CThread*	T, u32 THID)
{
	if (strstr(Core.Params, "-use_std"))
	{
		std_threads.push_back(

			new std::thread([&](CThread* thread)
				{
					CThread* T = dynamic_cast<CThread*>(thread);
					if (T)
					{
						if (T->thMessages)
							clMsg("*STD THREAD #%d: Started.", T->thID);

						DWORD_PTR affinityMask = 1ull << T->thID;
						HANDLE threadHandle = GetCurrentThread();
						SetThreadAffinityMask(threadHandle, affinityMask);

						T->Execute();
						T->thCompleted = TRUE;

						if (T->thMessages)
							clMsg("*STD THREAD #%d: Task Completed.", T->thID);
					}
				},
				T
			)

		);
	}
	else
	{
 		R_ASSERT(T);
		threads.push_back(T);
		T->Start();
	}
}

void	CThreadManager::wait	(u32	sleep_time)
{
	// Wait for completition
	char		perf			[1024];
  
	if (!std_threads.empty())
	{
 		for (int i = 0; i < std_threads.size(); i++)
 			std_threads[i]->join();
 
		for (int i = 0; i < std_threads.size(); i++)
			xr_delete(std_threads[i]);

		std_threads.clear();
	}
	else if (!threads.empty())
	{
 		for (;;)
		{
			Sleep(sleep_time);

			perf[0] = 0;
			float	sumProgress = 0;
			float	sumPerformance = 0;
			u32		sumComplete = 0;
			for (u32 ID = 0; ID < threads.size(); ID++)
			{
				sumProgress += threads[ID]->thProgress;
				sumComplete += threads[ID]->thCompleted ? 1 : 0;
				sumPerformance += threads[ID]->thPerformance;

				char				P[64];
				if (ID)
					xr_sprintf(P, "*%3.1f", threads[ID]->thPerformance);
				else
					xr_sprintf(P, " %3.1f", threads[ID]->thPerformance);

				xr_strcat(perf, P);
			}

			if (threads[0]->thMonitor)
			{
				Status("Performance: %3.1f :%s", sumPerformance, perf);
			}
			Progress(sumProgress / float(threads.size()));
			if (sumComplete == threads.size())
				break;
		}

		// Delete threads
		for (u32 thID = 0; thID < threads.size(); thID++)
			if (threads[thID]->thDestroyOnComplete)
				xr_delete(threads[thID]);
		threads.clear();
	}
 
}
