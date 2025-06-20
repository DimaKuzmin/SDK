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

	int ThreadsNum = 16;

	bool SkipRaytracing = false;
	bool Silent = false;
	bool Embree = true;
	bool EmbreeBVHCompact = false;
	bool EmbreeBVHRobust = false;
	bool ClearTemp = false;
	bool SkipTHM = false;

	bool AI = false;
	bool DO = false;
	bool LC = false;

	bool LC_Dxt1Avail = false;
  	bool LC_NoSun = false;
	bool LC_NoHemi = false;
	bool LC_NoRGB = false;
 
	bool LC_NoSMG = true;
	bool LC_Noise = false;
	bool LC_Tess = true;
	bool LC_SkipInvalidFaces = true;
	bool LC_tex_rgba = false;
	bool LC_NoSubdivide = false;
	bool LC_skipWeld = false;

	bool  LC_lmaps_alternative = false;
	int   LC_sizeLmaps = 1024 * 4;
	float LC_lmaps_max_pixels = 0.95f;

	bool IsOverloadedSettings = false;
	int LC_JSampleMU = 6;
	int LC_JSample = 9;
	float LC_Pixels = 10;

	float WeldDistance = 0.005f;

	bool DO_NoSun = false;
	int  DO_Samples = 7;

	// SPAWN COMPILER
	bool AI_BuildSpawn = false;

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
};
void RenderMainUI();
void RenderCompilerUI(int X, int Y);
void InitializeUIData(); 

extern CompilersMode gCompilerMode;