#include "CompilersUI.h"
#include "cl_log.h"

#include <timeapi.h>
#include "../Editors/XrLC/xrLC.h"
#include "../Editors/XrAI/xrAI.h"

#pragma comment(lib, "embree4.lib")
#pragma comment(lib, "tbb12.lib")
#pragma comment(lib, "XrSE_Factory.lib")

void setup_luabind_allocator();

CompilersMode gCompilerMode;

void SDL_Application();

#include <thread>

#define AI_COMPILER

// #include "../Editors/XrAI/xr_graph_merge.h"
// #include "../Editors/XrAI/game_spawn_constructor.h"
//  
// #include "../Editors/XrAI/game_graph_builder.h"
// #include "../Editors/XrAI/spawn_patcher.h"

#include "../Editors/XrAI/factory_api.h"
  

extern SEFactory_Create* create_entity = 0;
extern SEFactory_Destroy* destroy_entity = 0;

#include "../Editors/XrSE_Factory/xrSE_Factory_import_export.h"
extern "C"
{
	//FACTORY_API	ISE_Abstract* __stdcall create_entity(LPCSTR section);
	//FACTORY_API	void		__stdcall destroy_entity(ISE_Abstract*& abstract);
	FACTORY_API void		__stdcall initialize_factory();
	FACTORY_API void		__stdcall destroy_factory();
};

static HMODULE hFactory;

void InitialFactory() {
	LPCSTR g_name = "xrSE_Factory.dll";
	Msg("Loading DLL: %s", g_name);
	hFactory = LoadLibraryA(g_name);

	if (0 == hFactory)
		R_CHK(GetLastError());

	R_ASSERT2(hFactory, "Factory DLL raised exception during loading or there is no factory DLL at all");

	create_entity = (SEFactory_Create*) create_entity;
 
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


extern void MyLogCallback(const char* string);
extern void ProcessUISave(bool Save);
 
void StartCompile()
{
	clMsg("Run New Compilation !");

	ProcessUISave(true);
	
	// Give a LOG-thread a chance to startup
	std::thread
	(
		[]()
		{
			SetThreadDescription(GetCurrentThread(), L"Startup Compiler Thread");
			GetIterationData().clear();
			GetLogVector().clear();

			GetIterationData().push_back({ "xrLC" });
			GetIterationData().push_back({ "xrAI" });
			GetIterationData().push_back({ "xrDO" });

			auto InitilizeIteration = [](LCBuildingType Type, bool active, LPCSTR phase)
				{
					SetActiveIteration(&(GetIterationData()[(int)Type]));
					gCompilerMode.builder_type = Type;
					if (active)
					{
						GetActiveIteration()->status = InProgress;

						u32 dwTime = timeGetTime();
						Phase(phase);

						if (Type == LCBuildingType::eLC)
							MainCompilerLC();
  						else if (Type == LCBuildingType::eDO)
							MainCompilerDO();
						else if (Type == LCBuildingType::eAI)
						{
							setup_luabind_allocator();
							StartupAI();
						}

						dwTime = (timeGetTime() - dwTime) / 1000;

						GetActiveIteration()->status = Complited;
						GetActiveIteration()->elapsed_time = dwTime;
					}
					else
						GetActiveIteration()->status = Skip;

					PhasesEnd();
				};

			InitilizeIteration(LCBuildingType::eLC, gCompilerMode.LC, "xrLC Startup");
			InitilizeIteration(LCBuildingType::eAI, gCompilerMode.AI, "xrAI Startup");
 			InitilizeIteration(LCBuildingType::eDO, gCompilerMode.DO, "xrDO Startup");
			 
			// Show statistic
			extern xr_string make_time(u32 sec);
			for (auto& I : GetIterationData())
			{
				for (auto& PH : I.phases)
					clMsg("* %40s  : Time elapsed %s", PH.PhaseName.c_str(), make_time(PH.elapsed_time));

				clMsg("* Compiler (%s) : Time elapsed: %s ", I.iterationName.c_str(), make_time(I.elapsed_time));
			}
 			
			PhasesEnd();
		}
	).detach(); 

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
	ProcessUISave(false);

	initialize_factory();
	InitialFactory();


	InitializeUIData();
	SDL_Application();
	destroy_factory();
	return 0;
}