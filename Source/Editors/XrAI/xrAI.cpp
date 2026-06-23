// xrAI.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "../../xrcore/xr_ini.h"
#include "process.h"
#include "xrAI.h"

#include "xr_graph_merge.h"
#include "game_spawn_constructor.h"

#include "game_graph_builder.h"
#include <mmsystem.h>
#include "spawn_patcher.h"


extern LPCSTR LEVEL_GRAPH_NAME;
extern void	xrCompiler			(LPCSTR name, bool draft_mode, bool pure_covers, LPCSTR out_name);
 
extern void	verify_level_graph	(LPCSTR name, bool verbose);
   
string_path_ai INI_FILE;

extern  HWND logWindow;

extern LPCSTR GAME_CONFIG;

extern void clear_temp_folder	();
 
void StartupAI	()
{   
	// Load project
	for (auto& [Name, Selected] : gCompilerMode.Files)
	{
		if (!Selected)			continue;

		string4096 name;
		strcpy(name, Name.data());
		if (xr_strlen(name))
			xr_strcat(name, "\\");

		string_path prjName;
		prjName[0] = 0;
		bool can_use_name = false;

		if (xr_strlen(name) < sizeof(string_path))
		{
			can_use_name = true;
			FS.update_path(prjName, "$game_levels$", name);
		}

		FS.update_path(INI_FILE, "$game_config$", GAME_CONFIG);

		if (gCompilerMode.AI_BuildLevel)
		{
			R_ASSERT3(can_use_name, "Too big level name", name);

			char* output = (pstr)LEVEL_GRAPH_NAME;

			xrCompiler(prjName, gCompilerMode.AI_Draft, gCompilerMode.AI_PureCovers, output);
		}

		if (gCompilerMode.AI_Verify)
		{
			R_ASSERT3(can_use_name, "Too big level name", name);
			verify_level_graph(prjName, gCompilerMode.AI_Verbose);
		}
	}

	if (gCompilerMode.AI_BuildSpawn)
	{
		if (gCompilerMode.AI_Spawn_By_Freemp || gCompilerMode.AI_Spawn_SingleLevel)
		{

			for (auto& [Name, Selected] : gCompilerMode.Files)
			{
				if (!Selected) continue;

				FS.get_path("$level$")->_set(Name.c_str());
				gCompilerMode.set_level_name( Name.c_str() );

				std::thread([&Name]
					{
					clear_temp_folder();
					CGameSpawnConstructor(Name.c_str());
				}).join();
				
			} 			
		}
 		else
		{
			xr_string Levels;
			for (auto& [Name, Selected] : gCompilerMode.Files)
			{
				if (!Selected)	continue;
 				if (!Levels.empty()) Levels += ",";
				Levels += Name;
			}
 
			std::thread([&Levels]
				{
					clear_temp_folder();
					CGameSpawnConstructor(Levels.c_str());
				}).join();
			// CGameSpawnConstructor(Levels.c_str());
		}
	}

} 
