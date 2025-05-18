// xrLC.cpp : Defines the entry point for the application.
//
#include "stdafx.h"

#include "math.h"
#include "build.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "xrLC.h"

#include "../XrLCLight/xrDeflector.h"

CBuild*	pBuild		= NULL;
u32		version		= 0;

extern void logThread(void *dummy);
extern volatile BOOL bClose;
CTimer	dwStartupTime;
 
static const char* h_str =
"The following keys are supported / required:\n"
"-? or -h	== this help\n"
"-o			== modify build options\n"
"-nosun		== disable sun-lighting\n"
"-skipinvalid\t== skip crash if invalid faces exists\n"
"-notess	== don`t use tesselate geometry\n"
"-nosubd	== don`t use subdivide geometry\n"
"-tex_rgba	== don`t compress lightmap textures\n"
"-f<NAME>	== compile level in GameData\\Levels\\<NAME>\\\n"
"\n"
"NOTE: The last key is required for any functionality\n";

void Help(const char*);

typedef int __cdecl xrOptions(b_params* params, u32 version, bool bRunBuild);
extern bool g_using_smooth_groups;

extern CompilersMode gCompilerMode;

void MainCompilerLC()
{
	g_build_options.b_radiosity = false; // Более не подерживается
	g_build_options.b_noise = gCompilerMode.LC_Noise;
	g_using_smooth_groups = !gCompilerMode.LC_NoSMG;

	// Faster FPU 
	SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);

	// Load project
	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)
			continue;

		create_global_data();

		lc_global_data()->b_nohemi_set(gCompilerMode.LC_NoRGB);
		lc_global_data()->b_nohemi_set(gCompilerMode.LC_NoHemi);
		lc_global_data()->b_nosun_set(gCompilerMode.LC_NoSun);


		string256 temp;
		xr_sprintf(temp, "%s - Levels Compiler", Name.data());
		// SDL_SetWindowTitle(g_AppInfo.Window, temp);

		string_path prjName;
		FS.update_path(prjName, "$game_levels$", strconcat(sizeof(prjName), prjName, Name.data(), "\\build.prj"));

		string256 phaseName;
		Phase(strconcat(sizeof(phaseName), phaseName, "Reading project [", Name.data(), "]..."));

		string256 inf;
		IReader* F = FS.r_open(prjName);
		if (NULL == F)
		{
			xr_sprintf(inf, "Build failed!\nCan't find level: '%s'", Name.data());
			clMsg(inf);
			MessageBoxA(nullptr, inf, "Error!", MB_OK | MB_ICONERROR);
			return;
		}

		// Version
		F->r_chunk(EB_Version, &version);
		clMsg("version: %d", version);
		R_ASSERT(XRCL_CURRENT_VERSION == version);

		// Header
		b_params Params;
		F->r_chunk(EB_Parameters, &Params);

		// Conversion
		Phase("Converting data structures...");
		pBuild = new CBuild();
		pBuild->Load(Params, *F);

		gCompilerMode.scene_bbox = pBuild->scene_bb;

		if (gCompilerMode.IsOverloadedSettings)
		{
			g_params().m_lm_jitter_samples = gCompilerMode.LC_JSample;
			g_params().m_lm_pixels_per_meter = gCompilerMode.LC_Pixels;
			g_params().m_weld_distance = gCompilerMode.WeldDistance;
			setLMSIZE(gCompilerMode.LC_sizeLmaps);
		}

		FS.r_close(F);

		// Call for builder
		string_path lfn;
		FS.update_path(lfn, _game_levels_, Name.data());

		lc_global_data()->level_path = lfn;
 		clMsg("* LEVEL PATH: %s", lfn);

		pBuild->Run(lfn);
		xr_delete(pBuild);
	}
}


/*
void Startup( )
{
	create_global_data();

	std::string name = build_args->level_name.c_str();

	// Se7Kills ADD NEW Name Reading
	clMsg("LevelName: %s", name);

	extern  HWND logWindow;
	string256				temp;
	xr_sprintf(temp, "%s - Levels Compiler", name.c_str());
	SetWindowText(logWindow, temp);

	string_path				prjName;
	FS.update_path(prjName, "$game_levels$", strconcat(sizeof(prjName), prjName, name.c_str(), "\\build.prj"));

	string256				phaseName;
	Phase(strconcat(sizeof(phaseName), phaseName, "Reading project [", name.c_str(), "]..."));

	string256 inf;
	IReader* F = FS.r_open(prjName);
	if (NULL == F)
	{
		xr_sprintf(inf, "Build failed!\nCan't find level: '%s'", name);
		clMsg(inf);
		MessageBox(logWindow, inf, "Error!", MB_OK | MB_ICONERROR);
		return;
	}

	// Version
	F->r_chunk(EB_Version, &version);
	clMsg("version: %d", version);
	R_ASSERT(XRCL_CURRENT_VERSION == version);

	// Header
	b_params				Params;
	F->r_chunk(EB_Parameters, &Params);

	// Conversion
	Phase("Converting data structures...");
	pBuild = xr_new<CBuild>();
	pBuild->Load(Params, *F);
	 
	FS.r_close(F);

	// LOAD BUILD PARAMS
	g_params().m_lm_jitter_samples = args->sample;
	g_params().m_lm_pixels_per_meter = args->pxpm;

	g_build_options.b_noise = args->noise;
	lc_global_data()->b_nosun_set(args->nosun);
	lc_global_data()->b_norgb_set(args->norgb);
	lc_global_data()->b_nohemi_set(args->nohemi);


	// Call for builder
	string_path				lfn;
	dwStartupTime.Start();

	FS.update_path(lfn, _game_levels_, name.c_str());
	pBuild->Run(lfn);

	xr_delete(pBuild);

	// Show statistic
	u32	dwEndTime = dwStartupTime.GetElapsed_ms();
	xr_sprintf(inf, "Time elapsed: %s", make_time(dwEndTime / 1000).c_str());
	clMsg("Build succesful!\n%s", inf);

	Status("Построение Уровня Законечено! ");

	// Close log
	bClose = TRUE;
}

XRLC_API SpecialArgs* current_args_data = 0;
 
void Startup_DO(SpecialArgs* args)
{
	dwStartupTime.Start();

	// Give a LOG-thread a chance to startup
	InitCommonControls();
	thread_spawn(logThread, "log-update", 1024 * 1024, 0);
	Sleep(150);

	clMsg("Starting Thread Sturtup For Details Objects");

	// Load project
 	extern  HWND logWindow;
	string256			temp;
	xr_sprintf(temp, "%s - Detail Compiler", args->level_name);
	SetWindowText(logWindow, temp);

	//FS.update_path	(name,"$game_levels$",name);
	FS.get_path("$level$")->_set(args->level_name.c_str());

	CTimer				dwStartupTime;
	dwStartupTime.Start();
	 
	xrCompileDO(args->DoSamples);

	// Show statistic
	char	stats[256];
 	xr_sprintf(stats, "Time elapsed: %s", make_time((dwStartupTime.GetElapsed_ms()) / 1000).c_str());
	clMsg(stats);

	bClose = TRUE;
	Status("Построение Уровня Законечено! ");
}


#include <ctime>
#include "../XrLCLight/xrDeflector.h"
 

void ReadArgs(SpecialArgsXRLCLight* build_args, SpecialArgs* args)
{
	build_args->no_invalide_faces = args->no_invalide_faces;

	build_args->pxpm = args->pxpm;
	build_args->mu_samples = args->mu_samples;
	build_args->sample = args->sample;
	build_args->use_threads = args->use_threads;

	build_args->nohemi = args->nohemi;
	build_args->norgb = args->norgb;
	build_args->noise = args->noise;
	build_args->nosun = args->nosun;
	build_args->nosmg = args->nosmg;

	build_args->no_optimize = args->no_optimize;
	build_args->no_simplify = args->no_simplify;
		
	build_args->use_avx = args->use_avx;
	build_args->use_embree = args->use_embree;
	build_args->use_sse = args->use_sse;

 	build_args->level_name = args->level_name;

	build_args->skip_weld = args->skip_weld;

	build_args->use_DXT1 = args->use_DXT1;
  
 	build_args->run_mu_first = args->run_mu_first;

	build_args->EmbreeGeomType = args->EmbreeGeomType;
	build_args->useRobust = args->useRobust;
	build_args->useCompactEmbreeBVH = args->useCompactEmbreeBVH;

	build_args->LmapsHemi = args->LmapsHemi;
}
 
XRLC_API void StartupWorking(SpecialArgs* args)
{
	if (args->IsDOLighting)
	{
		build_args = new SpecialArgsXRLCLight();
		ReadArgs(build_args, args);

		Debug._initialize(false);
		Core._initialize("xrDO");
		Startup_DO(args);
		Core._destroy();
		return;
	}

	char tmp[256];
  
	sprintf(tmp, "c++: SCENE SET: PXPM: %f, SAMPLES: %u, MUSAMPLES: %u, threads: %u, SkipWeld: %u",
		args->pxpm, args->sample, args->mu_samples, args->use_threads, args->skip_weld);
	clMsg(tmp);

	sprintf(tmp, "c++: LIGHT SET: nohemi: %d, norgb: %d, nosun: %d, noise: %d, nosmg: %d",
		args->nohemi, args->norgb, args->nosun, args->noise, args->nosmg);
	clMsg(tmp);

	sprintf(tmp, "c++: xrLC SET:  Level: %s,  no_optimize: %d, no_simplify: %d, ",
		args->level_name.c_str(), args->no_optimize, args->no_simplify);
	clMsg(tmp);

	sprintf(tmp, "c++: EMBREE SET: USE embree: %d, avx: %d, sse: %d, use DXT1: %d",
		args->use_embree, args->use_avx, args->use_sse, args->use_DXT1);
	clMsg(tmp);

	current_args_data = args;

	switch (args->LightmapSize_enum)
	{
		case SpecialArgs::eLightmap1024:
		{
			setLMSIZE(1024);
		}break;
		case SpecialArgs::eLightmap2048:
		{
			setLMSIZE(2048);
		}break;

		case SpecialArgs::eLightmap4096:
		{
			setLMSIZE(4096);
		}break;

		case SpecialArgs::eLightmap8192:
		{
			setLMSIZE(8192);  
		}break;

	};
 
	build_args = new SpecialArgsXRLCLight();
	ReadArgs(build_args, args);
 
 	g_using_smooth_groups = args->nosmg;
	Startup(args);


	Core._destroy();
}

void SaveIni(CInifile* save, SpecialArgs* args)
{
	save->w_string("launcher", "level_name", args->level_name.c_str());

	save->w_bool("launcher", "use_compacted_bvh", args->useCompactEmbreeBVH);
	save->w_bool("launcher", "use_skip_invalid", args->no_invalide_faces);
	save->w_bool("launcher", "use_robust", args->useRobust);
	save->w_bool("launcher", "use_DXT1", args->use_DXT1);

	save->w_bool("launcher", "use_embree", args->use_embree);
	save->w_bool("launcher", "use_avx", args->use_avx);
	save->w_bool("launcher", "use_sse", args->use_sse);

	save->w_bool("launcher", "no_optimize", args->no_optimize);
	save->w_bool("launcher", "no_simplify", args->no_simplify);
	
	save->w_bool("launcher", "nosun", args->nosun);
	save->w_bool("launcher", "nohemi", args->nohemi);
	save->w_bool("launcher", "norgb", args->norgb);

	save->w_bool("launcher", "noise", args->noise);
	save->w_bool("launcher", "nosmg", args->nosmg);
	save->w_bool("launcher", "skip_weld", args->skip_weld);

 	save->w_bool("launcher", "run_mu_first", args->run_mu_first);


	save->w_u8("launcher", "EmbreeGeomType", args->EmbreeGeomType);
  
	save->w_u8("launcher", "LightmapSize", args->LightmapSize_enum);
 
	save->w_u32("launcher", "threads", args->use_threads);
	save->w_float("launcher", "pxpm", args->pxpm);
	save->w_u32("launcher", "sample", args->sample);
	save->w_u32("launcher", "mu_samples", args->mu_samples); 
}

void LoadIni(CInifile* load, SpecialArgs* args)
{
	args->level_name = load->r_string("launcher", "level_name");
	
	load->r_bool("launcher", "use_compacted_bvh");
	load->r_bool("launcher", "use_skip_invalid", args->no_invalide_faces);
	load->r_bool("launcher", "use_robust", args->useRobust);
	load->r_bool("launcher", "use_DXT1", args->use_DXT1);
	load->r_bool("launcher", "use_embree", args->use_embree);
	load->r_bool("launcher", "use_avx", args->use_avx);
	load->r_bool("launcher", "use_sse", args->use_sse);
	load->r_bool("launcher", "no_optimize", args->no_optimize);
	load->r_bool("launcher", "no_simplify", args->no_simplify);
	load->r_bool("launcher", "nosun", args->nosun);
	load->r_bool("launcher", "nohemi", args->nohemi);
	load->r_bool("launcher", "norgb", args->norgb);
	load->r_bool("launcher", "noise", args->noise);
	load->r_bool("launcher", "nosmg", args->nosmg);
	load->r_bool("launcher", "skip_weld", args->skip_weld);
	 
 	load->r_bool("launcher", "run_mu_first", args->run_mu_first);
	load->r_u8("launcher", "EmbreeGeomType", (u8&) args->EmbreeGeomType);
	load->r_u8("launcher", "LightmapSize", (u8&) args->LightmapSize_enum);
 
	load->r_u32("launcher", "threads", (u32&)args->use_threads);
	load->r_float("launcher", "pxpm", (float&) args->pxpm);
	load->r_u32("launcher", "sample", (u32&) args->sample);
	load->r_u32("launcher", "mu_samples", (u32&) args->mu_samples);
}

bool LoadParrams(SpecialArgs* args)
{
 	Debug._initialize(false);
	Core._initialize("xrLC");

	string_path fsPath;
	FS.update_path(fsPath, "$app_data_root$", "compiler_se7.ltx");
	
	bool isLoaded = false;
	CInifile* Reader = xr_new< CInifile >(fsPath, true, true);
	if (Reader && Reader->section_exist("launcher"))
	{
		isLoaded = true;
		LoadIni(Reader, args);
	}
	return isLoaded;
}

void SaveParrams(SpecialArgs* args)
{
	string_path fsPath;
	FS.update_path(fsPath, "$app_data_root$", "compiler_se7.ltx");
	CInifile* file = new CInifile(fsPath, false, false);
	if (file)
	{
		SaveIni(file, args);
	}
	file->save_as();
}
*/
