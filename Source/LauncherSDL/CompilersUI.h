#pragma once
#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>
#include "../XrCore/xrCore.h"  


struct ImFont;

struct LevelFileData
{
	xr_string Name;
	bool Select = false;
};

struct CompilersMode
{
	Fbox scene_bbox;

	int ThreadsNum = 8;

	bool SkipRaytracing = false;
	bool Silent = false;
	bool Embree = false;
	bool CUDA   = true;

	bool EmbreeBVHCompact = false;
	bool EmbreeBVHRobust = false;
	bool ClearTemp = false;
	bool SkipTHM = false;

	bool use_avx2 = false;
 
	bool AI = false;
	bool DO = false;
	bool LC = false;

  	bool LC_Cforms = true;
	bool LC_Dxt1Avail = false;
  	bool LC_NoSun = false;
	bool LC_NoHemi = false;
	bool LC_NoRGB = false;
 
	bool LC_NoSMG = true;
	bool LC_Noise = true;
	bool LC_Tess = true;
	bool LC_SkipInvalidFaces = true;
	bool LC_tex_rgba = false;
	bool LC_NoSubdivide = false;
	bool LC_skipWeld = false;

 	int  LC_lmap_size	= 1024 * 8;
	int  LC_lmap_BORDER = 1;
	float  LC_lmap_fill	= 0.89f;
	
	bool IsOverloadedSettings = true;
	int LC_JSampleMU = 6;
	int LC_JSample = 9;
	float LC_Pixels = 10;

	float WeldDistance = 0.005f;

	bool DO_NoSun = false;
	int  DO_Samples = 7;

	// SPAWN COMPILER
	bool AI_BuildSpawn			= false;
	bool AI_Map_NoLimits		= false;	
 	char AI_spawn_name[256];
	char AI_StartActor[256];
	bool AI_NoSeparatorCheck = true;

	bool AI_BuildLevel = false;
	bool AI_PureCovers = false;
	bool AI_Draft = false;
	bool AI_Verify = false;
	bool AI_Verbose = false;

	xr_vector<LevelFileData> Files;
	ImFont* CompilerIconsFont;

	shared_str LevelName;
	LPCSTR get_level_name()
	{
		return *LevelName;
	}
};
void RenderMainUI();
void RenderCompilerUI(int X, int Y);
void InitializeUIData(); 

extern CompilersMode gCompilerMode;