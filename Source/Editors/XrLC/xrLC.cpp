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
 
extern CompilersMode gCompilerMode;


void MainCompilerLC()
{
	// Load project
	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)	continue;

		gCompilerMode.LevelName = Name.data();
		create_global_data();

		string256 temp;
		xr_sprintf(temp, "%s - Levels Compiler", Name.data());
		// SDL_SetWindowTitle(g_AppInfo.Window, temp);

		string_path prjName;
		FS.update_path(prjName, "$game_levels$", strconcat(sizeof(prjName), prjName, Name.data(), "\\build.prj"));

		string256 inf;
		IReader* F = FS.r_open(prjName);
		if (NULL == F)
		{
			xr_sprintf(inf, "Build failed!\nCan't find level: '%s'", Name.data());
			MessageBoxA(nullptr, inf, "Error!", MB_OK | MB_ICONERROR);
			return;
		}

		// Version
		unsigned int version;
		F->r_chunk(EB_Version, &version);
		R_ASSERT(XRCL_CURRENT_VERSION == version);

		// Header
		b_params Params;
		F->r_chunk(EB_Parameters, &Params);

		// Conversion
		pBuild = new CBuild();
		pBuild->Load(Params, *F);

		gCompilerMode.scene_bbox = pBuild->scene_bb;

		if (gCompilerMode.IsOverloadedSettings)
		{
			g_params().m_lm_jitter_samples = gCompilerMode.LC_JSample;
			g_params().m_lm_pixels_per_meter = gCompilerMode.LC_Pixels;
			g_params().m_weld_distance = gCompilerMode.LC_WeldDistance;
		}

		FS.r_close(F);

		// Call for builder
		string_path lfn;
		FS.update_path(lfn, "$game_levels$", Name.data());

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

		xrCompileDO();

		// Show statistic
		char	stats[256];
		xr_sprintf(stats, "Time elapsed: %s", make_time((dwStartupTime.GetElapsed_ms()) / 1000).c_str());
		clMsg(stats);

		Status ("Построение Уровня Законечено! ");
	}

	bClose = TRUE;
	
}
