#pragma once
//#include "stdafx.h"

struct XRLC_LIGHT_API  SpecialArgsXRLCLight
{
	bool use_embree;
	bool use_avx;
	bool use_sse;
	bool use_opcode_old;


	int use_threads;


	bool no_optimize;
	bool invalide_faces;

	bool nosun;
	bool norgb;
	bool nohemi;

	bool no_simplify;
	bool noise;
	bool nosmg;


	float pxpm;
	int sample; // 1-9
	int mu_samples; // 1-6

	char* special_args;
};


extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;