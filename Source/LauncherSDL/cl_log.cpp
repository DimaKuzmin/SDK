#include "../XrCore/xrCore.h"
#include <time.h>
#include <mmsystem.h>
#include <CommCtrl.h>
#include "cl_log.h"

#include <Psapi.h>
#pragma comment(lib, "Psapi.lib")
 
// extern ILogger* LoggerCL = 0;

//************************* Log-thread data
static xrCriticalSection	csLog;
xr_vector<xr_string> myLogVector;
void MyLogCallback(const char* string) 
{
	csLog.Enter();
	myLogVector.push_back(string);
	csLog.Leave();
}

xr_vector<xr_string>& GetLogVector()
{
	return myLogVector;
}

static char					status	[1024	]	="";
static char					phase	[1024	]	="";

static u32					phase_start_time	= 0;
static BOOL					bStatusChange		= FALSE;
static BOOL					bPhaseChange		= FALSE;
static u32					phase_total_time	= 0;
 
xr_string make_time	(u32 sec)
{
	char		buf[64];
	xr_sprintf		(buf,"%2.0d:%2.0d:%2.0d",sec/3600,(sec%3600)/60,sec%60);
	int len		= int(xr_strlen(buf));
	for (int i=0; i<len; i++) if (buf[i]==' ') buf[i]='0';
	return xr_string(buf);
}
  
void Status	(const char *format, ...)
{
 	va_list				mark;
	va_start			( mark, format );
	vsprintf			( status, format, mark );
	bStatusChange		= TRUE;
	Msg					("    | $ %s",status);
 
}

void StatusNoMSG(const char* format, ...)
{
 	va_list				mark;
	va_start(mark, format);
	vsprintf(status, format, mark);
	bStatusChange = TRUE;
}
 
IterationData* ActiveIteration = nullptr;

size_t GetHeapMemory()
{
	PROCESS_MEMORY_COUNTERS_EX pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
	{
 		return pmc.PrivateUsage;
	}
};

void Phase			(const char *phase_name)
{
  	phase_total_time = timeGetTime() - phase_start_time;
	 
	// Start _new phase
	if (ActiveIteration->phases.size() > 0)
	{
  		ActiveIteration->phases[ActiveIteration->phases.size() - 1].used_memory = GetHeapMemory();
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].status = Complited;
	}

	ActiveIteration->phases.push_back({ phase_name });

	phase_start_time = timeGetTime();
	Progress(0);

	Msg("\n* New phase started: %s", phase_name);

 	Memory.mem_compact();
	log_vminfo();
}

void PhasesEnd()
{
	for (auto I : GetIterationData())
	{
		if (I.phases.size() > 0)
		{
			I.phases[I.phases.size() - 1].used_memory = GetHeapMemory();
 			I.phases[I.phases.size() - 1].status	  = Complited;
		}


		// Start _new phase
		// if (ActiveIteration->phases.size() > 0)
		// {
		// 	ActiveIteration->phases[ActiveIteration->phases.size() - 1].used_memory = GetHeapMemory();
		// 	ActiveIteration->phases[ActiveIteration->phases.size() - 1].status = Complited;
		// }
	}

}

extern CTimer	dwStartupTime;
  
static bool isLogCallback = false;

void clLog(const char* msg )
{
	if (!isLogCallback)
	{
		SetLogCB(MyLogCallback);
		isLogCallback = true;
	}

 	Log				(msg);
}

void clMsg( const char *format, ...)
{
	if (!isLogCallback)
	{
		SetLogCB(MyLogCallback); 
		isLogCallback = true;
	}

	va_list		mark;
	char buf	[4*256];
	va_start	( mark, format );
	vsprintf	( buf, format, mark );
 
	string1024		_out_;
	strconcat		(sizeof(_out_),_out_,"    |    | ", buf );
	clLog			(_out_);

}

static char					additional_data[1024] = "";

xr_vector<IterationData> iterationData;


xr_vector<IterationData>& GetIterationData()
{
	return iterationData;
}

IterationData* GetActiveIteration()
{
	return ActiveIteration;
}
void SetActiveIteration(IterationData* i)
{
	if (auto* p = (ActiveIteration ? &ActiveIteration->phases : nullptr);
		p && p->size() > 0 && (*p)[p->size() - 1].status != Complited)
		(*p)[p->size() - 1].status = Complited;

	ActiveIteration = i;
}

void AditionalData(const char* format, ...)
{
	va_list		mark;
	va_start(mark, format);
	vsprintf(additional_data, format, mark);
	
	csLog.Enter();
	if (ActiveIteration->phases.size() > 0)
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].AdditionalData = additional_data;
	csLog.Leave();
}


u32& GetPhaseStartTime()
{
	return phase_start_time;
}

#include <atomic>
std::atomic<float> progress = 0.0f;
void Progress(const float F)
{
	progress.store(F);

	// No critical section usage
 	if (ActiveIteration->phases.size() > 0)
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].PhasePersent = progress;
}

void ProgressMT(const float F)
{
	// No critical section usage
	progress.store(F);

	if (ActiveIteration->phases.size() > 0)
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].PhasePersent = progress.load();
}


float GetProgress()
{
	return progress;
}