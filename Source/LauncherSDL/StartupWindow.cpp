
#include "CompilersUI.h"
#include "cl_log.h"


#include <timeapi.h>
#include "../Editors/XrLC/xrLC.h"
// #include "../Editors/XrAI/xrAI.h"

void setup_luabind_allocator();

void Help(const char* h_str) {
	MessageBoxA(0, h_str, "Command line options", MB_OK | MB_ICONINFORMATION);
}

CompilersMode gCompilerMode;

extern bool ShowMainUI;
void Startup(LPSTR lpCmdLine)
{
	GetIterationData().push_back({ "xrLC" });
	GetIterationData().push_back({ "xrAI" });
	GetIterationData().push_back({ "xrDO" });

	u32 dwStartupTime = timeGetTime();

	SetActiveIteration(&(GetIterationData()[0]));
	u32 dwTimeLC = 0;

	if (gCompilerMode.LC)
	{
		GetActiveIteration()->status = InProgress;
		dwTimeLC = timeGetTime();
		Phase("xrLC Startup");
		MainCompilerLC();

		dwTimeLC = (timeGetTime() - dwTimeLC) / 1000;

		GetActiveIteration()->status = Complited;
		GetActiveIteration()->elapsed_time = dwTimeLC;
	}
	else
	{
		GetActiveIteration()->status = Skip;
	}

	SetActiveIteration(&(GetIterationData()[1]));
	u32 dwTimeAI = 0;
	if (gCompilerMode.AI)
	{
		GetActiveIteration()->status = InProgress;

		dwTimeAI = timeGetTime();
		Phase("xrAI Startup");

		setup_luabind_allocator();
		// StartupAI();
 
		dwTimeAI = (timeGetTime() - dwTimeAI) / 1000;

		GetActiveIteration()->status = Complited;
		GetActiveIteration()->elapsed_time = dwTimeLC;
	}
	else
	{
		GetActiveIteration()->status = Skip;
	}

	SetActiveIteration(&(GetIterationData()[2]));
	u32 dwTimeDO = 0;
	if (gCompilerMode.DO) {
		GetActiveIteration()->status = InProgress;
		dwTimeDO = timeGetTime();
		Phase("xrDO Startup");
		
		MainCompilerDO();
		dwTimeDO = (timeGetTime() - dwTimeDO) / 1000;

		GetActiveIteration()->status = Complited;
		GetActiveIteration()->elapsed_time = dwTimeLC;
	}
	else
	{
		GetActiveIteration()->status = Skip;
	}

	// Show statistic
	string256 stats;
	extern xr_string make_time(u32 sec);
	u32 dwEndTime = timeGetTime();

	xr_sprintf(
		stats,
		"Time elapsed: %s \r\n xrLC: %s\r\n xrAI: %s\r\n xrDO: %s",
		make_time((dwEndTime - dwStartupTime) / 1000).c_str(),
		make_time(dwTimeLC).c_str(),
		make_time(dwTimeAI).c_str(),
		make_time(dwTimeDO).c_str()
	);

	if (!gCompilerMode.Silent)
	{
		MessageBoxA(nullptr, stats, "Congratulation!", MB_OK | MB_ICONINFORMATION);
	}

	extern volatile BOOL bClose;

	// Close log
	bClose = TRUE;
 	ShowMainUI = true;
	Sleep(200);
}

void StartCompile()
{
	// Give a LOG-thread a chance to startup
	//	InitCommonControls();
	Sleep(150);
	thread_spawn(logThread, "log-update", 1024 * 1024, 0);
}

void SDL_Application();
int APIENTRY WinMain
(
	HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nCmdShow
)
{
	// Initialize debugging
	Debug._initialize(false);
	Core._initialize("X-Ray 1.8 Compilers");

	InitializeUIData();
	SDL_Application();

	return 0;
}
