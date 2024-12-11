#pragma once
//#include "stdafx.h"

struct XRLC_LIGHT_API  SpecialArgsXRLCLight
{
	enum EmbreeGeom
	{
		eLow = 0,
		eMiddle = 1,
		eHigh = 2,
		eRefit = 3
	};
 
	u64 RaysCalculated = 0;

	// debuging 
	bool off_lmaps = false;
	bool off_impl = false;
	bool off_mulitght = false;
	bool off_raytrace = false;

	bool use_DXT1 = false;
	bool use_fast_lmapsbuilder = false;
	

	// XRLC ADVANCED SETTINGS
	float embree_tnear = 0.2f;
	bool use_MethodIntersection = 1;

	int embree_geometry_type = EmbreeGeom::eLow;

	bool use_embree = 0;			//+
	bool use_tbb = 0;
	 
	bool use_avx = 0;				//+
	bool use_sse = 0;				//+
	bool use_opcode_old = 0;		//+
	bool use_RobustGeom = 0;		//+

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
	bool use_std = 0;				//+

	float pxpm = 10;				//+
	int sample = 9; // 1-9			//+
	int mu_samples = 6; // 1-6		//+
	int MaxHitsPerRay = 256;
 
	char* special_args = 0;
	
	bool use_cdbPacking = false;
	bool run_mu_first = false;
	bool useUV_v2 = false;

	std::string level_name;
};


extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;