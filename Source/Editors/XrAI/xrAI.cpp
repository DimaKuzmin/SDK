// xrAI.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "../../xrcore/xr_ini.h"
#include "process.h"
#include "xrAI.h"

#include "xr_graph_merge.h"
#include "game_spawn_constructor.h"
#include "game_spawn_unpacker.h"

#include "xrCrossTable.h"
//#include "path_test.h"
#include "game_graph_builder.h"
#include <mmsystem.h>
#include "spawn_patcher.h"


extern LPCSTR LEVEL_GRAPH_NAME;

extern void	xrCompiler			(LPCSTR name, bool draft_mode, bool pure_covers, LPCSTR out_name);
extern void logThread			(void *dummy);
extern volatile BOOL bClose;
extern void test_smooth_path	(LPCSTR name);
extern void test_hierarchy		(LPCSTR name);
extern void	xrConvertMaps		();
extern void	test_goap			();
extern void	smart_cover			(LPCSTR name);
extern void	verify_level_graph	(LPCSTR name, bool verbose);
//extern void connectivity_test	(LPCSTR);
extern void compare_graphs		(LPCSTR level_name);
extern void test_levels			();

static const char* h_str = 
	"The following keys are supported / required:\n"
	"-? or -h   == this help\n"
	"-f<NAME>   == compile level in gamedata/levels/<NAME>/\n"
	"-o         == modify build options\n"
	"-s         == build game spawn data\n"
	"\n"
	"NOTE: The last key is required for any functionality\n";

void Help()
{	MessageBox(0,h_str,"Command line options",MB_OK|MB_ICONINFORMATION); }

string_path_ai INI_FILE;

extern  HWND logWindow;

extern LPCSTR GAME_CONFIG;

extern void clear_temp_folder	();

SpecialArgsAI xrAI_Args;

void execute	(LPSTR cmd)
{
	/// Msg("Execute");
	/*
	char* unpack = strstr(cmd, "-unpack");
	char* out = strstr(cmd, "-out");

	if (unpack && out)
	{
		string256 spawn_unpack, spawn_out;
		sscanf(unpack + 7, "%s", spawn_unpack);
		sscanf(out + 4, "%s", spawn_out);

		game_spawn_unpacker(spawn_unpack, spawn_out);
		return;
	}
	*/
   
	// Load project
	string4096 name;
	xr_strcpy(name, xrAI_Args.level_name.c_str());

	if (xr_strlen(name))
		xr_strcat			(name,"\\");

	string_path			prjName;
	prjName				[0] = 0;
	bool				can_use_name = false;

 	Msg("LevelARGS: %s", xrAI_Args.level_name.c_str());
	Msg("OutARGS: %s", xrAI_Args.OutSpawn_Name.c_str());
	Msg("StartARGS: %s", xrAI_Args.SpawnActorStart.c_str());
	
	if (xr_strlen(name) < sizeof(string_path)) 
	{
		can_use_name	= true;
		FS.update_path	(prjName,"$game_levels$", name);
	}

	FS.update_path		(INI_FILE,"$game_config$", GAME_CONFIG);


	if (!xrAI_Args.UseSpawnCompiler) 
	{
		R_ASSERT3		(can_use_name,"Too big level name",name);
 		xrCompiler		(prjName, xrAI_Args.Draft, xrAI_Args.PureCovers, LEVEL_GRAPH_NAME);

		if (strstr(cmd, "-verify"))
		{
			R_ASSERT3(can_use_name, "Too big level name", name);
			verify_level_graph(prjName, !strstr(cmd, "-noverbose"));
		}
	}
	else
	{
 		if (xr_strlen(name))
			name[xr_strlen(name) - 1] = 0;

		LPCSTR START_LEVEL = xrAI_Args.SpawnActorStart.c_str();
		LPCSTR OUTSPAWN = xrAI_Args.OutSpawn_Name.c_str();

		clear_temp_folder();
		CGameSpawnConstructor(name, OUTSPAWN, START_LEVEL, xrAI_Args.NoSeparator);
  	}
}

extern XRAI_API ILoggerAI* LoggerCL_xrAI = 0;

void Startup(LPSTR     lpCmdLine)
{
	string4096 cmd;
 
	xr_strcpy(cmd,lpCmdLine);
	strlwr(cmd);
     
	// Give a LOG-thread a chance to startup
	InitCommonControls	();
	Sleep				(150);
	thread_spawn		(logThread,	"log-update", 1024*1024,0);
	while				(!logWindow)	Sleep		(150);
	
	u32					dwStartupTime	= timeGetTime();
	execute				(cmd);
	// Show statistic
	char				stats[256];
	extern				std::string make_time(u32 sec);
	extern				HWND logWindow;
	u32					dwEndTime = timeGetTime();
	xr_sprintf				(stats,"Time elapsed: %s",make_time((dwEndTime-dwStartupTime)/1000).c_str());
	MessageBox			(logWindow,stats,"Congratulation!",MB_OK|MB_ICONINFORMATION);

	bClose				= TRUE;
	FlushLog			();
	Sleep				(500);
}

#include "factory_api.h"

#include "quadtree.h"
#include "..\XrSE_Factory\xrSE_Factory_import_export.h"

void buffer_vector_test		();

XRAI_API void  StartupWorking_xrAI(SpecialArgsAI* args)
{
	xrAI_Args = *args;
}


/*
int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow)
{
	Debug._initialize		(false);
	Core._initialize		("xrai",0);
	XrSE_Factory::initialize();
	buffer_vector_test		();



	Startup					(lpCmdLine);

	XrSE_Factory::destroy();
	Core._destroy			();

	return					(0);
}
*/