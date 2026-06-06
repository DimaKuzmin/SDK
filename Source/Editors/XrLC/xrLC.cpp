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
 
extern CompilersMode gCompilerMode;

extern int lmapNameID;
void MainCompilerLC()
{
	clMsg("MainCompilerLC");
 	// Load project
	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)	continue;

		clMsg("Create GlobalData");
		gCompilerMode.LevelName = Name.data();
		create_global_data();
		lmapNameID = 0; // —брасываем ID карты !

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

		clMsg("Create Build file");

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

		Phase("Cleanup Data ...");

		Msg("[Memory] Start Removing Build: %llu mb", (GetHeapMemory() / 1024 / 1024) );
 		xr_delete(pBuild);
 		Memory.mem_compact();
 		Msg("[Memory] Start Removing Build: %llu mb", (GetHeapMemory() / 1024 / 1024));
	}
}

void MainCompilerDO()
{
  	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)			continue;
		FS.get_path("$level$")->_set(Name.c_str());
		xrCompileDO();
	}
}
