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

enum class LCBuildingType
{
	eNone = -1,
	eLC = 0,
	eAI = 1,
	eDO = 2
};

struct CompilersMode
{
	LCBuildingType builder_type = LCBuildingType::eNone;;
	// ComboBox Values;
	int RadioID = 0;
	int item_current_jitter = 2;
	int item_current_jitter_mu = 6;
	int item_cuda_rays = 1;
	int item_lmap_selected = 1;

	Fbox scene_bbox;

	int ThreadsNum = 8;

 	bool Silent = false;
	bool Embree = false;
	bool CUDA   = true;
	int	 LC_CUDA_RAYS_SIZE = 8192;

	bool EmbreeBVHCompact = false;
	bool EmbreeBVHRobust = false;
	bool ClearTemp = false;
	bool SkipTHM = false;
 	bool use_avx2 = false;
 
	bool AI = false;
	bool DO = false;
	bool LC = false;

 	bool LC_Dxt1Avail = false;
  	bool LC_NoSun = false;
	bool LC_NoHemi = false;
	bool LC_NoRGB = false;
 
	bool LC_NoSMG = true;
	bool LC_MakeProgressive = true;
	bool LC_MakeStriptify = true;

	bool LC_Tangent = true;
	bool LC_Tess = true;

	bool LC_RemoveInvalidFaces = false;
	bool LC_SkipInvalidFaces = true;
	bool LC_skipWeld = false;

	bool LC_Se7kills_method = true;
 	int  LC_lmap_size	= 1024 * 8;
	int  LC_lmap_BORDER = 1;
	float  LC_lmap_fill	= 0.89f;
	
	bool IsOverloadedSettings = true;
	int LC_JSampleMU = 6;
	int LC_JSample = 9;
	float LC_Pixels = 10;

	float LC_WeldDistance = 0.005f;

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