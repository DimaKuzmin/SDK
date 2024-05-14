#pragma once

#ifdef XRLC_API_EXPORTS
#	define XRLC_API __declspec(dllexport)
#else
#	define XRLC_API __declspec(dllimport)
#endif

#include <string>
  
struct XRLC_API  SpecialArgs
{
	enum EmbreeGeom
	{
		eLow = 0,
		eMiddle = 1,
		eHigh = 2,
		eRefit = 3
	};

	// debuging 
	bool off_lmaps = false;
	bool off_impl = false;
	bool off_mulitght = false;
	bool use_DXT1 = false;


	// XRLC ADVANCED SETTINGS
	float embree_tnear = 0.2f;
	int embree_geometry_type = EmbreeGeom::eLow;

	bool use_embree = 0;
	bool use_avx = 0;
	bool use_sse = 0;
	bool use_opcode_old = 0;
	bool use_RobustGeom = 0;


	bool use_IMPLICIT_Stage = 0;
	bool use_LMAPS_Stage = 0;
	bool use_MU_Lighting = 0;

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
	bool use_std = 0;


	float pxpm = 10;
	int sample = 9; // 1-9
	int mu_samples = 6; // 1-6
	int MaxHitsPerRay = 256;
	bool MU_ModelsRegression = true;

	char* special_args = 0;
	std::string level_name;
};

XRLC_API void  StartupWorking(SpecialArgs* args);

class XRLC_API ILogger
{
public:
	virtual void  updateLog(LPCSTR str) = 0;
	virtual void  updatePhrase(LPCSTR phrase) = 0;
	virtual void  updateStatus(LPCSTR status) = 0;

	virtual void  UpdateText() = 0;
	virtual void  UpdateTime(LPCSTR time) = 0;
};

extern XRLC_API ILogger* LoggerCL;