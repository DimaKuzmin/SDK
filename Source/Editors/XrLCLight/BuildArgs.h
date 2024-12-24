#pragma once
//#include "stdafx.h"

struct XRLC_LIGHT_API  SpecialArgsXRLCLight
{
	enum LightmapSize
	{
		eLightmap1024 = 0,
		eLightmap2048 = 1,
		eLightmap4096 = 2,
		eLightmap8192 = 3
	};
	// debuging 
	bool use_DXT1 = false;

	// XRLC ADVANCED SETTINGS

	bool use_embree = 0;			//+
	 
	bool use_avx = 0;				//+
	bool use_sse = 0;				//+

	int use_threads = 4;			//+


	bool no_optimize = 0;			//+-
	bool no_invalide_faces = 0;		//+

	bool nosun = 0;					//+
	bool norgb = 0;					//+
	bool nohemi = 0;				//+

	bool no_simplify = 0;			//+ 
	bool noise = 0;					//+
	bool nosmg = 0;					//+
	bool skip_weld = 0;				//+

	float pxpm = 10;				//+
	int sample = 9; // 1-9			//+
	int mu_samples = 6; // 1-6		//+
  
	char* special_args = 0;
	
 	bool run_mu_first = false;

	std::string level_name;
};


extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;