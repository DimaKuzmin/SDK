// xrLC.cpp : Defines the entry point for the application.
//
#include "stdafx.h"

#include "math.h"
#include "build.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "xrLC.h"

//#pragma comment(linker,"/STACK:0x800000,0x400000")
//#pragma comment(linker,"/HEAP:0x70000000,0x10000000")

#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

#define PROTECTED_BUILD

#ifdef PROTECTED_BUILD
#	define TRIVIAL_ENCRYPTOR_ENCODER
#	define TRIVIAL_ENCRYPTOR_DECODER
#	include "../../xrEngine/trivial_encryptor.h"
#	undef TRIVIAL_ENCRYPTOR_ENCODER
#	undef TRIVIAL_ENCRYPTOR_DECODER
#endif // PROTECTED_BUILD

CBuild*	pBuild		= NULL;
u32		version		= 0;

extern void logThread(void *dummy);
extern volatile BOOL bClose;
  

typedef int __cdecl xrOptions(b_params* params, u32 version, bool bRunBuild);

CTimer	dwStartupTime;

void Startup(LPSTR     lpCmdLine, SpecialArgs* args)
{
	create_global_data();
	char cmd[512],name[256];
	BOOL bModifyOptions		= FALSE;

	xr_strcpy(cmd,lpCmdLine);
	strlwr(cmd);
 	
	// Give a LOG-thread a chance to startup
	//_set_sbh_threshold(1920);
	InitCommonControls		();
	thread_spawn			(logThread, "log-update",	1024*1024,0);
	Sleep					(150);
	
	// Faster FPU 
	SetPriorityClass		(GetCurrentProcess(),NORMAL_PRIORITY_CLASS);

 

	log_vminfo();
	
	// Load project
	name[0]=0;				
	//sscanf(strstr(cmd,"-f")+2,"%s",name);

	// Se7Kills ADD NEW Name Reading
	xr_strcpy(name, build_args->level_name.c_str() );
	clMsg("LevelName: %s", name);

	extern  HWND logWindow;
	string256				temp;
	xr_sprintf				(temp, "%s - Levels Compiler", name);
	SetWindowText			(logWindow, temp);

 

	string_path				prjName;
	FS.update_path			(prjName,"$game_levels$",strconcat(sizeof(prjName),prjName,name,"\\build.prj"));
	string256				phaseName;
	Phase					(strconcat(sizeof(phaseName),phaseName,"Reading project [",name,"]..."));
 
	string256 inf;
	IReader*	F			= FS.r_open(prjName);
	if (NULL==F)
	{
		xr_sprintf				(inf,"Build failed!\nCan't find level: '%s'",name);
		clMsg				(inf);
		MessageBox			(logWindow,inf,"Error!",MB_OK|MB_ICONERROR);
		return;
	}
 
	// Version
	F->r_chunk			(EB_Version,&version);
	clMsg				("version: %d",version);
	R_ASSERT(XRCL_CURRENT_VERSION==version);

	// Header
	b_params				Params;
	F->r_chunk			(EB_Parameters,&Params);

	// Show options if needed
	/* 
	if (bModifyOptions)		
	{
		Phase		("Project options...");
		HMODULE		L = LoadLibrary		("xrLC_Options.dll");
		void*		P = GetProcAddress	(L,"_frmScenePropertiesRun");
		R_ASSERT	(P);
		xrOptions*	O = (xrOptions*)P;
		int			R = O(&Params,version,false);
		FreeLibrary	(L);
		if (R==2)	{
			ExitProcess(0);
		}
	}
	*/
 
	// Conversion
	Phase					("Converting data structures...");
	pBuild					= xr_new<CBuild>();
	pBuild->Load			(Params,*F);
	 
	FS.r_close				(F);

	// LOAD BUILD PARAMS
	g_params().m_lm_jitter_samples = args->sample;
	g_params().m_lm_pixels_per_meter = args->pxpm;

	g_build_options.b_noise = args->noise;
	lc_global_data()->b_nosun_set(args->nosun);
	lc_global_data()->b_norgb_set(args->norgb);
	lc_global_data()->b_nohemi_set(args->nohemi);
	

	// Call for builder
	string_path				lfn;
	
	
	dwStartupTime.Start();

	FS.update_path			(lfn,_game_levels_,name);
	pBuild->Run				(lfn);
	xr_delete				(pBuild);

	// Show statistic
	extern	std::string make_time(u32 sec);
	u32	dwEndTime			= dwStartupTime.GetElapsed_ms();
	xr_sprintf					(inf,"Time elapsed: %s",make_time(dwEndTime/1000).c_str());
	clMsg					("Build succesful!\n%s",inf);

	//if (!strstr(cmd,"-silent"))
	//	MessageBox			(logWindow,inf,"Congratulation!",MB_OK|MB_ICONINFORMATION);

	Status("Построение Уровня Законечено! ");

	// Close log
	bClose					= TRUE;
	Sleep					(500);
}

//typedef void DUMMY_STUFF (const void*,const u32&,void*);
//XRCORE_API DUMMY_STUFF	*g_temporary_stuff;
//XRCORE_API DUMMY_STUFF	*g_dummy_stuff;



#include <ctime>


void ReadArgs(SpecialArgsXRLCLight* build_args, SpecialArgs* args)
{
	build_args->no_invalide_faces = args->no_invalide_faces;

	build_args->pxpm = args->pxpm;
	build_args->mu_samples = args->mu_samples;
	build_args->sample = args->sample;
	build_args->use_threads = args->use_threads;


	build_args->nohemi = args->nohemi;
	build_args->norgb = args->norgb;
	build_args->noise = args->noise;
	build_args->nosun = args->nosun;
	build_args->nosmg = args->nosmg;

	build_args->no_optimize = args->no_optimize;
	build_args->no_simplify = args->no_simplify;

	build_args->use_avx = args->use_avx;
	build_args->use_embree = args->use_embree;
	build_args->use_sse = args->use_sse;
	build_args->use_opcode_old = args->use_opcode_old;

	build_args->special_args = args->special_args;
	build_args->level_name = args->level_name;

	build_args->embree_geometry_type = args->embree_geometry_type;
	build_args->use_RobustGeom = args->use_RobustGeom;
	build_args->skip_weld = args->skip_weld;
	build_args->embree_tnear = args->embree_tnear;
}

XRLC_API void StartupWorking(SpecialArgs* args)
{
	Debug._initialize(false);
	Core._initialize("xrLC");
 
	build_args = new SpecialArgsXRLCLight();
	ReadArgs(build_args, args);
 
 	g_using_smooth_groups = args->nosmg;
	Startup("", args);


	Core._destroy();
}


/*
 
int APIENTRY WinMain(HINSTANCE hInst,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow)
{
//	g_temporary_stuff	= &trivial_encryptor::decode;
//	g_dummy_stuff		= &trivial_encryptor::encode;

	// Initialize debugging
	Debug._initialize	(false);
	Core._initialize	("xrLC");
	
 
	Startup				(lpCmdLine);
	Core._destroy		();

	// Get the current time
	std::time_t currentTime = std::time(nullptr);

	// Convert the time to a string representation
	char* timeString = std::ctime(&currentTime);


	string_path pp;
	string128 tmp;
	sprintf(tmp, "xrLC_compile_log_%s.log", timeString);


	FS.update_path(pp, "$app_root$", tmp);
	
	IWriter * log_file_time = FS.w_open(pp);
	for (auto phase : *phases_timers_Get())
	{
		log_file_time->w_stringZ(phase.c_str());
 	}
	
	FS.w_close(log_file_time);
	 
	return 0;
}

*/