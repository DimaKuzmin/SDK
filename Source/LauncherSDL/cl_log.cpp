#include "../XrCore/xrCore.h"
#include <time.h>
#include <mmsystem.h>
#include <CommCtrl.h>
#include "cl_log.h"
 
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

volatile BOOL				bClose				= FALSE;

static char					status	[1024	]	="";
static char					phase	[1024	]	="";
static float				progress			= 0.0f;
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
	csLog.Enter			();
	va_list				mark;
	va_start			( mark, format );
	vsprintf			( status, format, mark );
	bStatusChange		= TRUE;
	Msg					("    | %s",status);

	csLog.Leave			();
}

void StatusNoMSG(const char* format, ...)
{
	csLog.Enter();
	va_list				mark;
	va_start(mark, format);
	vsprintf(status, format, mark);
	bStatusChange = TRUE;
 	csLog.Leave();
}
 
IterationData* ActiveIteration = nullptr;

void Progress		(const float F)
{
	// No critical section usage
	progress		= F;	

	if (ActiveIteration->phases.size() > 0)
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].PhasePersent = F;
}
 
void Phase			(const char *phase_name)
{
	csLog.Enter();
 	phase_total_time = timeGetTime() - phase_start_time;
	 

	// Start _new phase
	if (ActiveIteration->phases.size() > 0)
	{
		size_t  w_free, w_reserved, w_committed;
		vminfo(&w_free, &w_reserved, &w_committed);
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].used_memory = w_committed;
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].status = Complited;
	}

	ActiveIteration->phases.push_back({ phase_name });

	phase_start_time = timeGetTime();
	Progress(0);

	Msg("\n* New phase started: %s", phase_name);

 	Memory.mem_compact();
	log_vminfo();

	csLog.Leave();
}

extern CTimer	dwStartupTime;
 
void logThread(void *dummy)
{
	extern void Startup(LPSTR lpCmdLine);

	SetLogCB(MyLogCallback);

	string128 cmd;
	Startup(cmd);
 
	SetLogCB(0);
}
 
void clLog(const char* msg )
{
	csLog.Enter		();
	Log				(msg);
 	csLog.Leave();
}



void clMsg( const char *format, ...)
{
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
	csLog.Enter();

	va_list		mark;
	va_start(mark, format);
	vsprintf(additional_data, format, mark);


	if (ActiveIteration->phases.size() > 0)
	{
		ActiveIteration->phases[ActiveIteration->phases.size() - 1].AdditionalData = additional_data;
	}

	csLog.Leave();
}


u32& GetPhaseStartTime()
{
	return phase_start_time;
}


float GetProgress()
{
	return progress;
}