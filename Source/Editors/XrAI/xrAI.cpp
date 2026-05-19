// xrAI.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "../../xrcore/xr_ini.h"
#include "process.h"
#include "xrAI.h"

#include "xr_graph_merge.h"
#include "game_spawn_constructor.h"

#include "xrCrossTable.h"
#include "game_graph_builder.h"
#include <mmsystem.h>
#include "spawn_patcher.h"


extern LPCSTR LEVEL_GRAPH_NAME;
extern void	xrCompiler			(LPCSTR name, bool draft_mode, bool pure_covers, LPCSTR out_name);
 
extern volatile BOOL bClose;
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
		if (!Selected)
			continue;

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
		xr_string Levels;

		for (auto& [Name, Selected] : gCompilerMode.Files)
		{
			if (!Selected)
				continue;

			if (!Levels.empty())
				Levels += ",";

			Levels += Name;
		}

		string512 name = {};
		strcpy(name, Levels.data());
		if (xr_strlen(name))
			name[xr_strlen(name)] = 0;

		xr_string output = gCompilerMode.AI_spawn_name;
 		if (output.empty())
 			output = "new";
 
		char* start_level = gCompilerMode.AI_StartActor;
		if (!xr_strlen(start_level))
		{
			start_level = nullptr;
		}

		clear_temp_folder();

		clMsg("Processing : %s", name);
		CGameSpawnConstructor* BuilderSpawn = new CGameSpawnConstructor(name, output.data(), start_level, gCompilerMode.AI_NoSeparatorCheck);
	}

} 
