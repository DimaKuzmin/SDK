#pragma once

// #ifdef XRLC_API_EXPORTS
// #	define XRLC_API __declspec(dllexport)
// #else
// #	define XRLC_API __declspec(dllimport)
// #endif

void MainCompilerLC();
  
/*
#include <string>
 
struct XRLC_API SpecialArgs
{
	enum LightmapSize
	{
		eLightmap1024 = 0,
		eLightmap2048 = 1,
		eLightmap4096 = 2,
		eLightmap8192 = 3
	}; 

	unsigned int   EmbreeGeomType;
	bool		   useRobust;
	bool		   useCompactEmbreeBVH;

	LightmapSize LightmapSize_enum;
    
	// debuging 
	bool use_DXT1 = false;

	bool use_embree = 0;
	bool use_avx = 0;
	bool use_sse = 0;

	int use_threads = 4;


	bool no_optimize = 0;
	bool no_invalide_faces = 0;

	bool nosun = 0;
	bool norgb = 0;
	bool nohemi = 0;

	bool no_simplify = 0;
	bool noise = 0;
	bool nosmg = 0;
	bool skip_weld = 0;
 
	float pxpm = 10;
	int sample = 9; // 1-9
	int mu_samples = 6; // 1-6
 
 	bool run_mu_first = false;
	std::string level_name;

	bool IsDOLighting = false;
	unsigned int DoSamples = 0;

	// Debuging Functions
	bool LmapsComputation = true;
	bool LmapsHemi		  = false;
	bool adptive_ht    = true;
	bool cform_export  = true;
	 
};

XRLC_API void  StartupWorking(SpecialArgs* args);

extern XRLC_API SpecialArgs* current_args_data;

class XRLC_API ILogger
{
public:
	virtual void  updateLog(LPCSTR str) = 0;
	virtual void  updatePhrase(LPCSTR phrase) = 0;
	virtual void  updateStatus(LPCSTR status) = 0;
	virtual void  UpdateProgressBar(float value) = 0;

	virtual void  UpdateText() = 0;
	virtual void  UpdateTime(LPCSTR time, unsigned int time_global) = 0;
};

extern XRLC_API ILogger* LoggerCL;



extern XRLC_API bool LoadParrams(SpecialArgs* args);
extern XRLC_API void SaveParrams(SpecialArgs* args);
*/