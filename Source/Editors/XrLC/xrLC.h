#pragma once

#ifdef XRLC_API_EXPORTS
#	define XRLC_API __declspec(dllexport)
#else
#	define XRLC_API __declspec(dllimport)
#endif
 
struct XRLC_API  SpecialArgs
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

XRLC_API void  StartupWorking(LPSTR lpCmdLine, SpecialArgs* args);

class XRLC_API Logger
{
public:
	virtual void  updateLog(LPCSTR str) = 0;
	virtual void  updatePhrase(LPCSTR phrase) = 0;
};

extern XRLC_API Logger* LoggerCL;