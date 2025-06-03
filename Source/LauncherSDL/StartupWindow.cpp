
#include "CompilersUI.h"
#include "cl_log.h"


#include <timeapi.h>
#include "../Editors/XrLC/xrLC.h"
#include "../Editors/XrAI/xrAI.h"

void setup_luabind_allocator();

void Help(const char* h_str) {
	MessageBoxA(0, h_str, "Command line options", MB_OK | MB_ICONINFORMATION);
}

CompilersMode gCompilerMode;

extern bool ShowMainUI;
void StartupCompilers()
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
		StartupAI();
 
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

void SDL_Application();

#include <thread>

void StartCompile()
{
	// Give a LOG-thread a chance to startup
 	Sleep(150);
 	std::thread(logThread).detach();
}

#include "../Editors/XrAI/xrAI.h"

#define AI_COMPILER

#include "../Editors/XrAI/xr_graph_merge.h"
#include "../Editors/XrAI/game_spawn_constructor.h"
#include "../Editors/XrAI/xrCrossTable.h"
#include "../Editors/XrAI/game_graph_builder.h"
#include "../Editors/XrAI/spawn_patcher.h"

#include "../Editors/XrAI/factory_api.h"
  

extern SEFactory_Create* create_entity = 0;
extern SEFactory_Destroy* destroy_entity = 0;

static HMODULE hFactory;

void InitialFactory() {
	LPCSTR g_name = "xrSE_Factory.dll";
	Msg("Loading DLL: %s", g_name);
	hFactory = LoadLibraryA(g_name);

	if (0 == hFactory)
		R_CHK(GetLastError());

	R_ASSERT2(hFactory, "Factory DLL raised exception during loading or there is no factory DLL at all");

#ifdef _M_X64
	create_entity = (SEFactory_Create*) GetProcAddress(hFactory, "create_entity");	
	R_ASSERT(create_entity);
	destroy_entity = (SEFactory_Destroy*) GetProcAddress(hFactory, "destroy_entity");
	R_ASSERT(destroy_entity);
#else
	create_entity = (Factory_Create*)GetProcAddress(hFactory, "_create_entity@4");	R_ASSERT(create_entity);
	destroy_entity = (Factory_Destroy*)GetProcAddress(hFactory, "_destroy_entity@4");	R_ASSERT(destroy_entity);
#endif
}

void DestroyFactory() {
	FreeLibrary(hFactory);
}



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
	InitialFactory();


	InitializeUIData();
	SDL_Application();
	return 0;
}