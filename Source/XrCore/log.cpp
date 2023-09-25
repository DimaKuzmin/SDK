#include "stdafx.h"
#pragma hdrstop

#include <time.h>
#include "resource.h"
#include "log.h"
#if 0
	#include "malloc.h"
#endif

IWriter* filelog = nullptr;
IWriter* filelog_dump_data = nullptr;
 
extern BOOL					LogExecCB		= TRUE;
static string_path			logFName		= "engine.log";
static string_path			log_file_name	= "engine.log";
static BOOL 				no_log			= TRUE;
#ifdef PROFILE_CRITICAL_SECTIONS
	static xrCriticalSection	logCS(MUTEX_PROFILE_ID(log));
#else // PROFILE_CRITICAL_SECTIONS
	static xrCriticalSection	logCS;
#endif // PROFILE_CRITICAL_SECTIONS
xr_vector<shared_str>*		LogFile			= NULL;
static LogCallback			LogCB			= 0;

void ThreadLog()
{
	while (true)
	{
		if (filelog)
			filelog->flush();

		Sleep(500);
	}
}

void FlushLog			()
{
	//if (filelog)
	//	filelog->flush();
}

void AddOne				(const char *split) 
{
	if(!LogFile)		
						return;

	logCS.Enter			();

#ifdef DEBUG
	OutputDebugString	(split);
	OutputDebugString	("\n");
#endif

//	DUMP_PHASE;
	{
		shared_str			temp = shared_str(split);
//		DUMP_PHASE;
		LogFile->push_back	(temp);
	}

	//exec CallBack
	if (LogExecCB&&LogCB)
		LogCB(split);

	if (filelog)
		filelog->w_string(split);

	//FlushLog();

	logCS.Leave				();
}

void AddOne_fast(const char* split)
{
	logCS.Enter();
	if (filelog_dump_data)
		filelog_dump_data->w_string(split);
	logCS.Leave();
}

void Log				(const char *s) 
{
	int		i,j;

	u32			length = xr_strlen( s );
#if 1    
	PSTR split  = (PSTR)_alloca( (length + 1) * sizeof(char) );
#else
	PSTR split  = (PSTR)alloca( (length + 1) * sizeof(char) );
#endif
	for (i=0,j=0; s[i]!=0; i++) 
	{
		if (s[i]=='\n') {
			split[j]=0;	// end of line
			if (split[0]==0) { split[0]=' '; split[1]=0; }
			AddOne(split);
			j=0;
		} else {
			split[j++]=s[i];
		}
	}
	split[j]=0;
	AddOne(split);
}

void __cdecl Msg		( const char *format, ...)
{
	va_list		mark;
	string2048	buf;
	va_start	(mark, format );
	int sz		= _vsnprintf(buf, sizeof(buf)-1, format, mark ); buf[sizeof(buf)-1]=0;
    va_end		(mark);
	if (sz)		Log(buf);
}

void __cdecl Msg_IN_FILE(const char* format, ...)
{
	va_list		mark;
	string2048	buf;
	va_start(mark, format);
	int sz = _vsnprintf(buf, sizeof(buf) - 1, format, mark); buf[sizeof(buf) - 1] = 0;
	va_end(mark);
	if (sz)
	{
		AddOne_fast(buf);
	}
}


void Log				(const char *msg, const char *dop) {
	if (!dop) 
	{
		Log		(msg);
		return;
	}

	u32			buffer_size = (xr_strlen(msg) + 1 + xr_strlen(dop) + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );
	strconcat	(buffer_size, buf, msg, " ", dop);
	Log			(buf);
}

void Log				(const char *msg, u32 dop) {
	u32			buffer_size = (xr_strlen(msg) + 1 + 10 + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );

	xr_sprintf	(buf, buffer_size, "%s %d", msg, dop);
	Log			(buf);
}

void Log				(const char *msg, int dop) {
	u32			buffer_size = (xr_strlen(msg) + 1 + 11 + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );

	xr_sprintf	(buf, buffer_size, "%s %i", msg, dop);
	Log			(buf);
}

void Log				(const char *msg, float dop) {
	// actually, float string representation should be no more, than 40 characters,
	// but we will count with slight overhead
	u32			buffer_size = (xr_strlen(msg) + 1 + 64 + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );

	xr_sprintf	(buf, buffer_size, "%s %f", msg, dop);
	Log			(buf);
}

void Log				(const char *msg, const Fvector &dop) {
	u32			buffer_size = (xr_strlen(msg) + 2 + 3*(64 + 1) + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );

	xr_sprintf	(buf, buffer_size,"%s (%f,%f,%f)",msg, VPUSH(dop) );
	Log			(buf);
}

void Log				(const char *msg, const Fmatrix &dop)	{
	u32			buffer_size = (xr_strlen(msg) + 2 + 4*( 4*(64 + 1) + 1 ) + 1) * sizeof(char);
	PSTR buf	= (PSTR)_alloca( buffer_size );

	xr_sprintf	(buf, buffer_size,"%s:\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n%f,%f,%f,%f\n",
		msg,
		dop.i.x, dop.i.y, dop.i.z, dop._14_,
		dop.j.x, dop.j.y, dop.j.z, dop._24_,
		dop.k.x, dop.k.y, dop.k.z, dop._34_,
		dop.c.x, dop.c.y, dop.c.z, dop._44_
	);
	Log			(buf);
}

void LogWinErr			(const char *msg, long err_code)	{
	Msg					("%s: %s",msg,Debug.error2string(err_code)	);
}

LogCallback SetLogCB	(LogCallback cb)
{
	LogCallback	result	= LogCB;
	LogCB				= cb;
	return				(result);
}

LPCSTR log_name			()
{
	return				(log_file_name);
}

void InitLog()
{
	R_ASSERT			(LogFile==NULL);
	LogFile				= xr_new< xr_vector<shared_str> >();
	LogFile->reserve	(1000);
}

#include <thread>

void CreateLog			(BOOL nl)
{
    no_log				= nl;
	strconcat			(sizeof(log_file_name), log_file_name, Core.ApplicationName,"_",Core.UserName,".log");
	
	if (FS.path_exist("$logs$"))
		FS.update_path	(logFName,"$logs$", log_file_name);
	
	if (!no_log)
	{
        IWriter *f		= FS.w_open	(logFName);
        if (f==NULL)
		{
        	MessageBox	(NULL,"Can't create log file.","Error",MB_ICONERROR);
        	abort();
        }
        FS.w_close		(f);
    }

	filelog = FS.w_open(logFName);
	for (auto s : *LogFile)
	{
		filelog->w_string(s.c_str());
	}
	
	if (FS.path_exist("$logs$"))
	{
		string_path path;
		string128 name = {0};

		sprintf(name, "dump_data\\%s_%s.dump", Core.ApplicationName, Core.UserName);

		FS.update_path	(path, "$logs$", name);
	

		filelog_dump_data = FS.w_open(path);

		std::thread* th = new std::thread(ThreadLog);
		th->detach();
	}

}

void CloseLog(void)
{
	FlushLog		();
 	LogFile->clear	();
	xr_delete		(LogFile);
	FS.w_close(filelog);
	FS.w_close(filelog_dump_data);
}
