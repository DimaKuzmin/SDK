#include "stdafx.h"
#include "xrlight_implicitrun.h"
#include "xrThread.h"
#include "xrLight_Implicit.h"
#include "xrlight_implicitdeflector.h"
#include "BuildArgs.h"



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

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;
 
void RunThread(ImplicitDeflector& defl)
{
	CThreadManager			tmanager;
 
	u32	stride = defl.Height() / build_args->use_threads;
	
 	for (u32 thID = 0; thID < build_args->use_threads; thID++)
	{
		ImplicitThread* th = xr_new<ImplicitThread>(thID, &defl, thID * stride, thID * stride + stride);
		
		tmanager.start(th, thID);
	}

	tmanager.wait();
}	   

#ifndef DevCPU
#include "xrHardwareLight.h"
void RunCudaThread();
#endif


extern u64 results;
 
 

void RunImplicitMultithread(ImplicitDeflector& defl)
{
	// Start threads
#ifdef DevCPU
	RunThread(defl);
#else 
	if (!xrHardwareLight::IsEnabled())
		RunThread(defl);
	else
		RunCudaThread();
#endif

}

 