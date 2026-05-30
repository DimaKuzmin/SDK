
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

#pragma comment(lib, "XrSE_Factory.lib")

void SaveConfigFile(CInifile* file)
{
	string32 SECTION = "compiler_options";

	// values
 	// int
	file->w_s32(SECTION, "MaxThreads", gCompilerMode.ThreadsNum);
 	file->w_s32(SECTION, "lmap_border", gCompilerMode.LC_lmap_BORDER);
 
	file->w_s32(SECTION, "RAYS_ITEM", gCompilerMode.item_cuda_rays);
	file->w_s32(SECTION, "LMAP_ITEM", gCompilerMode.item_lmap_selected);
	file->w_s32(SECTION, "JITTER_ITEM", gCompilerMode.item_current_jitter_mu);
	file->w_s32(SECTION, "JITTER_MU_ITEM", gCompilerMode.item_current_jitter);
	file->w_s32(SECTION, "RADIO_SELECTOR", gCompilerMode.RadioID);


	//float
	file->w_float(SECTION, "JPixels", gCompilerMode.LC_Pixels);
	file->w_float(SECTION, "LC_WeldDistance", gCompilerMode.LC_WeldDistance);
	file->w_float(SECTION, "lmap_fill", gCompilerMode.LC_lmap_fill);


	// boolean
	
 	file->w_bool(SECTION, "Embree_Bool", gCompilerMode.Embree);
	file->w_bool(SECTION, "CUDA_Bool", gCompilerMode.CUDA);

   	file->w_bool(SECTION, "Embree_BVH_Compact", gCompilerMode.EmbreeBVHCompact);
	file->w_bool(SECTION, "Embree_BVH_Robust", gCompilerMode.EmbreeBVHRobust);
	file->w_bool(SECTION, "Embree_AVX", gCompilerMode.use_avx2);

	file->w_bool(SECTION, "ClearTempFiles", gCompilerMode.ClearTemp);
	file->w_bool(SECTION, "SkipTHM", gCompilerMode.SkipTHM);

 	file->w_bool(SECTION, "CompilerAI", gCompilerMode.AI);
	file->w_bool(SECTION, "CompilerDO", gCompilerMode.DO);
	file->w_bool(SECTION, "CompilerLC", gCompilerMode.LC);

	file->w_bool(SECTION, "LC_DXT1", gCompilerMode.LC_Dxt1Avail);
	file->w_bool(SECTION, "LC_NO_SUN", gCompilerMode.LC_NoSun);
	file->w_bool(SECTION, "LC_NO_HEMI", gCompilerMode.LC_NoHemi);
	file->w_bool(SECTION, "LC_NO_RGB", gCompilerMode.LC_NoRGB);
 	file->w_bool(SECTION, "LC_NO_SMG", gCompilerMode.LC_NoSMG);
	file->w_bool(SECTION, "LC_PROGRESSIVE_GEOM", gCompilerMode.LC_MakeProgressive);
	file->w_bool(SECTION, "LC_STRIPTIFY_GEOM", gCompilerMode.LC_MakeStriptify);
 	file->w_bool(SECTION, "LC_TANGENT_BASIS", gCompilerMode.LC_Tangent);
	file->w_bool(SECTION, "LC_TESSELATE", gCompilerMode.LC_Tess);
	file->w_bool(SECTION, "LC_SKIP_INVALID", gCompilerMode.LC_SkipInvalidFaces);
 	file->w_bool(SECTION, "LC_SKIP_WELD", gCompilerMode.LC_skipWeld);
 	file->w_bool(SECTION, "LC_LMAP_SE7", gCompilerMode.LC_Se7kills_method);
 
	// AI MAP
	file->w_bool(SECTION, "AI_BuildSpawn", gCompilerMode.AI_BuildSpawn);
	file->w_bool(SECTION, "AI_Map_NoLimits", gCompilerMode.AI_Map_NoLimits);
 	file->w_bool(SECTION, "AI_NoSeparatorCheck", gCompilerMode.AI_NoSeparatorCheck);
 	file->w_bool(SECTION, "AI_BuildLevel", gCompilerMode.AI_BuildLevel);
	file->w_bool(SECTION, "AI_PureCovers", gCompilerMode.AI_PureCovers);
	file->w_bool(SECTION, "AI_Draft",		gCompilerMode.AI_Draft);
	file->w_bool(SECTION, "AI_Verbose",		gCompilerMode.AI_Verbose);
	file->w_bool(SECTION, "AI_Verify",		gCompilerMode.AI_Verify);

	file->w_string(SECTION, "AI_SpawnName", gCompilerMode.AI_spawn_name);
	file->w_string(SECTION, "AI_StartActor", gCompilerMode.AI_StartActor);

	file->save_as();
	xr_delete(file);
}

void LoadConfigFile(CInifile* file)
{
	string32 SECTION = "compiler_options";
	if (!file->section_exist(SECTION)) return;
	
	Msg("Loading IniConfig: %s", file->fname());

	// values
	// int
	file->r_s32(SECTION, "MaxThreads", gCompilerMode.ThreadsNum);
 	file->r_s32(SECTION, "lmap_border", gCompilerMode.LC_lmap_BORDER);
 
	file->r_s32(SECTION, "RAYS_ITEM", gCompilerMode.item_cuda_rays);
	file->r_s32(SECTION, "LMAP_ITEM", gCompilerMode.item_lmap_selected);
	file->r_s32(SECTION, "JITTER_ITEM", gCompilerMode.item_current_jitter_mu);
	file->r_s32(SECTION, "JITTER_MU_ITEM", gCompilerMode.item_current_jitter);
	file->r_s32(SECTION, "RADIO_SELECTOR", gCompilerMode.RadioID);
 
	//float
	file->r_float(SECTION, "JPixels", gCompilerMode.LC_Pixels);
	file->r_float(SECTION, "LC_WeldDistance", gCompilerMode.LC_WeldDistance);
	file->r_float(SECTION, "lmap_fill", gCompilerMode.LC_lmap_fill);

	// boolean

	file->r_bool(SECTION, "Embree_Bool", gCompilerMode.Embree);
	file->r_bool(SECTION, "CUDA_Bool", gCompilerMode.CUDA);

	file->r_bool(SECTION, "Embree_BVH_Compact", gCompilerMode.EmbreeBVHCompact);
	file->r_bool(SECTION, "Embree_BVH_Robust", gCompilerMode.EmbreeBVHRobust);
	file->r_bool(SECTION, "Embree_AVX", gCompilerMode.use_avx2);

	file->r_bool(SECTION, "ClearTempFiles", gCompilerMode.ClearTemp);
	file->r_bool(SECTION, "SkipTHM", gCompilerMode.SkipTHM);

	file->r_bool(SECTION, "CompilerAI", gCompilerMode.AI);
	file->r_bool(SECTION, "CompilerDO", gCompilerMode.DO);
	file->r_bool(SECTION, "CompilerLC", gCompilerMode.LC);

	file->r_bool(SECTION, "LC_DXT1", gCompilerMode.LC_Dxt1Avail);
	file->r_bool(SECTION, "LC_NO_SUN", gCompilerMode.LC_NoSun);
	file->r_bool(SECTION, "LC_NO_HEMI", gCompilerMode.LC_NoHemi);
	file->r_bool(SECTION, "LC_NO_RGB", gCompilerMode.LC_NoRGB);
	file->r_bool(SECTION, "LC_NO_SMG", gCompilerMode.LC_NoSMG);
	file->r_bool(SECTION, "LC_PROGRESSIVE_GEOM", gCompilerMode.LC_MakeProgressive);
	file->r_bool(SECTION, "LC_STRIPTIFY_GEOM", gCompilerMode.LC_MakeStriptify);
	file->r_bool(SECTION, "LC_TANGENT_BASIS", gCompilerMode.LC_Tangent);
	file->r_bool(SECTION, "LC_TESSELATE", gCompilerMode.LC_Tess);
	file->r_bool(SECTION, "LC_SKIP_INVALID", gCompilerMode.LC_SkipInvalidFaces);
	file->r_bool(SECTION, "LC_SKIP_WELD", gCompilerMode.LC_skipWeld);
	file->r_bool(SECTION, "LC_LMAP_SE7", gCompilerMode.LC_Se7kills_method);


	// AI MAP
	file->r_bool(SECTION, "AI_BuildSpawn", gCompilerMode.AI_BuildSpawn);
	file->r_bool(SECTION, "AI_Map_NoLimits", gCompilerMode.AI_Map_NoLimits);
	file->r_bool(SECTION, "AI_NoSeparatorCheck", gCompilerMode.AI_NoSeparatorCheck);
	file->r_bool(SECTION, "AI_BuildLevel", gCompilerMode.AI_BuildLevel);
	file->r_bool(SECTION, "AI_PureCovers", gCompilerMode.AI_PureCovers);
	file->r_bool(SECTION, "AI_Draft", gCompilerMode.AI_Draft);
	file->r_bool(SECTION, "AI_Verbose", gCompilerMode.AI_Verbose);
	file->r_bool(SECTION, "AI_Verify", gCompilerMode.AI_Verify);
	
	// xr_strcpy( gCompilerMode.AI_spawn_name, file->r_string(SECTION, "AI_SpawnName") );
	// xr_strcpy( gCompilerMode.AI_StartActor,  file->r_string(SECTION, "AI_StartActor") );

	xr_delete(file);
}

void ProcessUISave(bool Save)
{
	string_path file;
	FS.update_path(file, "$app_data_root$", "UICompiler.ltx");

	if (!Save)
	{
  		LoadConfigFile(xr_new < CInifile >(file, true, true));
  	}
	else
	{
		SaveConfigFile(xr_new < CInifile >(file, false, false));
	}
}


void StartCompile()
{
	// Give a LOG-thread a chance to startup
	Sleep(150);
	std::thread(logThread).detach();

 	ProcessUISave(true);
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