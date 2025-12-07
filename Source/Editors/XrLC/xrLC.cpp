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
		}
		
		setLMSIZE(gCompilerMode.LC_sizeLmaps);

		FS.r_close(F);

		// Call for builder
		string_path lfn;
		FS.update_path(lfn, _game_levels_, Name.data());

		lc_global_data()->level_path = Name.data();
 		clMsg("* LEVEL PATH: %s", lfn);

		pBuild->Run(lfn);
		xr_delete(pBuild);
	}
}

void MainCompilerDO()
{
  	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)
			continue;
		FS.get_path("$level$")->_set(Name.c_str());

		CTimer				dwStartupTime;
		dwStartupTime.Start();

		xrCompileDO(gCompilerMode.DO_Samples);

		// Show statistic
		char	stats[256];
		xr_sprintf(stats, "Time elapsed: %s", make_time((dwStartupTime.GetElapsed_ms()) / 1000).c_str());
		clMsg(stats);

		Status ("Построение Уровня Законечено! ");
	}

	bClose = TRUE;
	
}


/*
    
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
