#include "app_info.h"

#pragma comment(lib, "Luabind.lib")
#pragma comment(lib, "lua51.lib")
#pragma comment(lib, "winmm.lib") 

#pragma comment(lib, "d3dx9.lib")
#pragma comment(lib, "SDL3.lib")
#pragma comment(lib, "FreeMagic.lib")
#pragma comment(lib, "BearCore.lib")
#pragma comment(lib, "BearGraphics.lib")

// Xray
#pragma comment(lib, "xrCore.lib")
#pragma comment(lib, "xrCDB.lib")

#pragma comment(lib, "xrLCLight.lib")
#pragma comment(lib, "xrLC.lib")
#pragma comment(lib, "xrAI.lib")
#pragma comment(lib, "xrDXT.lib")
 
CAppInfo g_AppInfo;

bool CAppInfo::IsSecondaryThread() const noexcept
{
	return false;
}

bool CAppInfo::IsPrimaryThread() const noexcept
{
	return true;
}

#include "CompilersUI.h"
void SaveConfigFile(CInifile* file)
{
	string32 SECTION = "compiler_options";

	// values
	// int
	file->w_s32(SECTION, "MaxThreads", gCompilerMode.ThreadsNum);
	file->w_s32(SECTION, "lmap_border", gCompilerMode.LC_lmap_BORDER);

	file->w_s32(SECTION, "RAYS_ITEM", gCompilerMode.item_cuda_rays);
	file->w_s32(SECTION, "LMAP_ITEM", gCompilerMode.item_lmap_selected);
	file->w_s32(SECTION, "JITTER_ITEM", gCompilerMode.item_current_jitter);
	file->w_s32(SECTION, "JITTER_MU_ITEM", gCompilerMode.item_current_jitter_mu);
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
	file->w_bool(SECTION, "AI_Draft", gCompilerMode.AI_Draft);
	file->w_bool(SECTION, "AI_Verbose", gCompilerMode.AI_Verbose);
	file->w_bool(SECTION, "AI_Verify", gCompilerMode.AI_Verify);

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
	file->r_s32(SECTION, "JITTER_ITEM", gCompilerMode.item_current_jitter);
	file->r_s32(SECTION, "JITTER_MU_ITEM", gCompilerMode.item_current_jitter_mu);
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