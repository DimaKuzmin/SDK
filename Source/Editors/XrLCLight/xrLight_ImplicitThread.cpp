#include "stdafx.h"
#include "xrlight_implicitrun.h"
#include "xrThread.h"
#include "xrLight_Implicit.h"
#include "xrlight_implicitdeflector.h"



class ImplicitThread : public CThread
{
public:

	ImplicitExecute		execute;
	ImplicitThread		(u32 ID, ImplicitDeflector* _DATA, u32 _y_start, u32 _y_end) : CThread (ID), execute( _y_start, _y_end, ID )
	{	
		if (ID == 0)
			execute.clear();
	}

	virtual void		Execute	();
};

void	ImplicitThread ::	Execute	()
{
	// Priority
	SetThreadPriority		(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	Sleep					(0);
	execute.Execute(0);
}



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

#define	NUM_THREADS	THREADS_COUNT()



 
void RunThread(ImplicitDeflector& defl)
{
	CThreadManager			tmanager;

	u32	stride = defl.Height() / NUM_THREADS;
	
 	for (u32 thID = 0; thID < NUM_THREADS; thID++)
	{
		ImplicitThread* th = xr_new<ImplicitThread>(thID, &defl, thID * stride, thID * stride + stride);
		
		tmanager.start(th);
	}

	tmanager.wait();
}	   

#include "xrHardwareLight.h"
extern u64 results;

void RunCudaThread();
void RunGPU_CPU();

void RunImplicitMultithread(ImplicitDeflector& defl)
{
	// Start threads
	
	if (strstr(Core.Params, "-hwcpu"))
	{
		 RunGPU_CPU();
	}
	else 
	{
	   	if (!xrHardwareLight::IsEnabled())
			RunThread(defl);
		else
			RunCudaThread();
	} 
}

 