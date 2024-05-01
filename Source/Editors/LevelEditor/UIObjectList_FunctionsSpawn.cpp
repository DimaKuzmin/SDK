#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "SpawnPoint.h"
#include "../xrServerEntities/xrServer_Objects_Alife_Smartcovers.h"
#include "../xrServerEntities/xrServer_Objects_ALife.h"

bool create_cfg_file = true;
bool select_file = false;
 
void UIObjectList::HideCombatCovers()
{
	ESceneCustomOTool* Spawn_Objects = dynamic_cast<ESceneCustomOTool*>(Scene->GetOTool(OBJCLASS_SPAWNPOINT));

	for (auto obj : Spawn_Objects->GetObjects())
	{
		CSpawnPoint* spawn = smart_cast<CSpawnPoint*>(obj);
		if (spawn)
		{
			CSE_SmartCover* smart_cover = smart_cast<CSE_SmartCover*>(spawn->m_SpawnData.m_Data);
			if (smart_cover && smart_cover->m_is_combat_cover)
				spawn->Show(false);
		}
	}
}
 
void UIObjectList::ClearCustomData()
{
	ESceneCustomOTool* tool = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

	if (tool)
	{
		for (auto obj : tool->GetObjects())
		{
			CSpawnPoint* sp = smart_cast<CSpawnPoint*>(obj);

			if (sp && sp->Selected())
			{
 				sp->m_SpawnData.ModifyCustomData("");
			}
		}
	}
}

bool custom_data_check(LPCSTR custom_data, xr_vector<LPCSTR>& vec)
{
	for (auto item : vec)
	{
		if (strstr(custom_data, item))
			return false;
	}

	return true;
};

void UIObjectList::CheckCustomData()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SPAWNPOINT));

	string256 file;
	sprintf(file, "CustomData\\%s.ltx", "custom_datas");

	string_path ignore;
	string_path path;
	FS.update_path(path, _import_, file);
	FS.update_path(ignore, _import_, "CustomData\\ignore_custom.ltx");

	CInifile* fileignore = xr_new<CInifile>(ignore, true);

	xr_vector<LPCSTR> ignore_vec;

	if (fileignore)
	{
		int id = 1;
		string128 line;
		sprintf(line, "ignore_%d", id);
		while (fileignore->line_exist("ignore", line))
		{
			LPCSTR name = fileignore->r_string("ignore", line);
			ignore_vec.push_back(name);

			id++;
			sprintf(line, "ignore_%d", id);

			Msg("Add To Ignore: %s", name);
		}
	}

	IWriter* w = FS.w_open(path);

	xr_vector<shared_str> list_file;

	ObjectList& list = ot->GetObjects();
	for (auto item : list)
	{
		CSpawnPoint* sp = (CSpawnPoint*)item;
		if (sp && sp->Selected())
		{
			shared_str custom_data = sp->m_SpawnData.ReadCustomData();

			if (custom_data.size() > 2 && custom_data_check(custom_data.c_str(), ignore_vec))
			{
				string512 tmp;
				sprintf(tmp, "[%s]\n %s\n\n\n", sp->GetName(), custom_data.c_str());

				list_file.push_back(tmp);
			}
		}
	}

	std::sort(list_file.begin(), list_file.end(), [](shared_str a, shared_str b) {return a.size() < b.size(); });

	if (w)
		for (auto str : list_file)
			w->w_string(str.c_str());

	FS.w_close(w);
}

void UIObjectList::UpdateCustomData()
{
	Msg("Update CustomData");
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SPAWNPOINT));

	customdata_objects.clear();


	ObjectList& list = ot->GetObjects();
	for (auto item : list)
	{
		CSpawnPoint* sp = (CSpawnPoint*)item;
		if (sp)
		{
			shared_str name = sp->m_SpawnData.ReadCustomData();
			if (name.size() > 2)
			{
				customdata_objects.push_back(item);
				Msg("Name:%s(%s), size: %d", item->GetName(), sp->m_SpawnData.m_Visual ? sp->m_SpawnData.m_Visual->source->visual_name.c_str() : "", name.size());
			}

		}
	}
}
 
void UIObjectList::ClearGraphs()
{
	ESceneCustomOTool* base = Scene->GetOTool(OBJCLASS_SPAWNPOINT);
	int i = 0;
	for (auto item : base->GetObjects())
	{
		CSpawnPoint* sp = (CSpawnPoint*)item;
		if (strstr(item->RefName(), "graph_point"))
		{
 			CSE_ALifeGraphPoint* graph = smart_cast<CSE_ALifeGraphPoint*>(sp->m_SpawnData.m_Data);

			if (graph)
			{
				graph->m_caConnectionPointName._set("");
				graph->m_tLocations[0] = 0;
				graph->m_tLocations[1] = 0;
				graph->m_tLocations[2] = 0;
				graph->m_tLocations[3] = 0;
				//Msg("Graph: %s, DATA: %u, %u, %u, %u, %s, %s", graph->name_replace(), graph->m_tLocations[0], graph->m_tLocations[1], graph->m_tLocations[2], graph->m_tLocations[3], graph->m_caConnectionLevelName.c_str(), graph->m_caConnectionPointName.c_str());
			}
		}
	}
}
 
void UIObjectList::GenSpawnCFG(xr_string section, xr_string map, xr_string prefix)
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SPAWNPOINT));

	ObjectList& list = ot->GetObjects();
	int i = 0;

	for (auto sel : list)
	{
		if (!sel->Selected())
			continue;

		string4096 text;
		sprintf(text, "[%s] \ncfg=scripts\\%s\\%s\\%s.ltx", section.c_str(), map.c_str(), prefix.c_str(), sel->GetName());

		string4096		buff;
		xr_sprintf(buff, sizeof(buff), "\"%s\"", (text) ? text : "");

		string128 t;
		sprintf(t, "scripts\\%s\\%s\\%s.ltx", map.c_str(), prefix.c_str(), sel->GetName());

		string_path p;
		FS.update_path(p, "$game_config$", t);

		CSpawnPoint* spawn = smart_cast<CSpawnPoint*>(sel);
		if (spawn)
		{
			IWriter* I = FS.w_open(p);
			I->w_string(spawn->m_SpawnData.ReadCustomData());
			FS.w_close(I);

			spawn->m_SpawnData.ModifyCustomData(buff);
		}
	}


}
 
void UIObjectList::CreateLogicConfigs()
{
	xr_string file_path;
	string_path ex = { 0 };

	if (EFS.GetOpenName(EDevice.m_hWnd, _import_spawns_, file_path))
	{
		CInifile* read_params = new CInifile(file_path.c_str(), false, true);

		xr_vector<bool> camps_ids;
		camps_ids.resize(256);
		u32 size_anims;

		LPCSTR smart_file;
		LPCSTR smart_prefix;
		LPCSTR include_sect_name;
		LPCSTR meet_sect_name;

		CInifile::Sect sect;
		CInifile::Sect meet_sect;

		if (read_params)
		{
			LPCSTR camps = read_params->r_string("animpoint", "camps");
			size_anims = read_params->r_u32("animpoint", "anims");

			include_sect_name = read_params->r_string("animpoint", "include_sect");
			meet_sect_name = read_params->r_string("animpoint", "meet_sect");

			sect = read_params->r_section(include_sect_name);
			meet_sect = read_params->r_section(meet_sect_name);



			smart_file = read_params->r_string("animpoint", "logic_cfg");
			smart_prefix = read_params->r_string("animpoint", "logic_prefix");



			u32 items = _GetItemCount(camps);
			for (int i = 0; i < items; i++)
			{
				string32 out_str;
				_GetItem(camps, i, out_str);
				int id = atoi(out_str);
				camps_ids[id] = true;
			}
		}

		xr_delete(read_params);

		CInifile* file = new CInifile(file_path.c_str(), false, false);
		/*
		{
			file->w_string("meet@global_meat", "close_anim", "nil");
			file->w_string("meet@global_meat", "close_victim", "nil");
			file->w_string("meet@global_meat", "far_anim", "nil");
			file->w_string("meet@global_meat", "far_victim", "nil");
			file->w_string("meet@global_meat", "close_distance", "0");
			file->w_string("meet@global_meat", "far_distance", "0");
			file->w_string("meet@global_meat", "use", "true");
			file->w_string("meet@global_meat", "snd_on_use", "nil");
			file->w_string("meet@global_meat", "meet_on_talking", "false");
		}


		LPCSTR section_include = "animpoint@global_animation";

		{
			file->w_string(section_include, "reach_movement", "walk_noweap");
			file->w_string(section_include, "meet", "meet@global_meat");
			file->w_string(section_include, "combat_ignore_cond", "true");
			file->w_string(section_include, "combat_ignore_keep_when_attacked", "false");
			file->w_string(section_include, "gather_items_enabled", "false");
			file->w_string(section_include, "help_wounded_enabled", "false");
			file->w_string(section_include, "corpse_detection_enabled", "false");
		}

		*/

		file->sections().push_back(&sect);
		file->sections().push_back(&meet_sect);

		for (int i = 0; i < size_anims; i++)
		{
			string32 logic;
			sprintf(logic, "logic@anim_%d", i);
			string32 active;
			sprintf(active, "animpoint@anim_%d", i);

			file->w_string(logic, "active", active);
			file->w_u32(logic, "prior", 100);
			file->w_string(logic, "suitable", "true");

			string32 cover_name;
			sprintf(cover_name, "%s_anim_%d", smart_prefix, i);

			file->w_string(active, "cover_name", cover_name);
			file->w_string(active, "use_camp", camps_ids[i] ? "true" : "false");
			file->w_section_include(active, sect.Name.c_str());
		}

		xr_strcat(ex, file_path.c_str());
		xr_strcat(ex, ".logic");
		file->save_as(ex);

		string_path smart_path = { 0 };
		xr_strcat(smart_path, file_path.c_str());
		xr_strcat(smart_path, ".smart");
		CInifile* ini_smart = new CInifile(smart_path, false, false);

		for (int i = 0; i < size_anims; i++)
		{
			string32 logic;
			sprintf(logic, "anim_%d", i);

			ini_smart->w_string("exclusive", logic, smart_file);
		}
		ini_smart->save_as(smart_path);
	}
}
 
void UIObjectList::SetCustomData(bool autoNumarate, LPCSTR logic, LPCSTR path_name)
{
	xr_string file;
	if (create_cfg_file && select_file)
	{
		if (EFS.GetOpenName(EDevice.m_hWnd, _import_spawns_, file))
		{
			//ini = new CInifile(file.c_str(), false, true);
		}
	}

	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SPAWNPOINT));

	ObjectList& list = ot->GetObjects();
	int i = 0;
	for (auto item : list)
	{
		if (item->Selected())
		{
			CSpawnPoint* sp = (CSpawnPoint*)item;

			if (sp && create_cfg_file || sp && select_file)
			{
				string4096 text = { 0 };
				sprintf(text, "[%s] \ncfg = scripts\\%s\\%s.ltx", logic, path_name, sp->GetName());

				string4096		buff;
				xr_sprintf(buff, sizeof(buff), "\"%s\"", (text) ? text : "");

				sp->m_SpawnData.ModifyCustomData(buff);

				string_path create_file = { 0 };
				string256 path_calc = { 0 };

				sprintf(path_calc, "scripts\\%s\\%s.ltx", path_name, sp->GetName());

				FS.update_path(create_file, _game_config_, path_calc);

				if (create_cfg_file && !select_file)
				{
					CInifile* file = new CInifile(create_file, false, false);

					file->w_u32("smart_terrain", "max_population", 1);
					file->w_u32("smart_terrain", "squad_id", i);
					string32 sec = { 0 };
					sprintf(sec, "respawn@sim_%d", i);
					file->w_string("smart_terrain", "respawn_params", sec);


					file->w_string(sec, "spawn_sim", "");

					char* squads[8] =
					{
						"sim_stalker_squad",  "sim_bandit_squad",   "sim_zombied_squad",
						"sim_monolith_squad", "sim_dolg_squad",     "sim_freedom_squad",
						"sim_killer_squad",   "sim_army_squad"//,     "sim_ecolog_squad"
					};

					char* monster_squads[15] =
					{
						"simulation_bloodsucker", "simulation_boar", "simulation_burer",
						"simulation_dog", "simulation_pseudodog", "simulation_flesh",
						"simulation_snork", "simulation_controller", "simulation_mix_dogs",
						"simulation_mix_boar_flesh", "simulation_chimera", "simulation_psy_dog",
						"simulation_tushkano", "simulation_gigant", "simulation_zombied"
					};

					bool monster = false;

					if (50 < Random.randI(0, 100))
						monster = true;


					if (!monster)
						file->w_string("spawn_sim", "spawn_squads", squads[Random.randI(1, 9)]);
					else
						file->w_string("spawn_sim", "spawn_squads", monster_squads[Random.randI(1, 15)]);

					file->w_u32("spawn_sim", "spawn_num", 3);

					file->save_as(create_file);
				}

				if (!file.empty() && select_file)
				{
					CInifile* ini = new CInifile(file.c_str(), false, true);
					if (ini->section_exist("exclusive"))
					{
						int l_count = ini->line_count("exclusive");

						for (int i = 0; i < l_count; i++)
						{
							LPCSTR name;
							LPCSTR value;
							ini->r_line("exclusive", i, &name, &value);

							string_path exclusive, final_path;
							//CALC PATH 1
							string256 file = { 0 };
							xr_strcat(file, "copy_logic\\");
							xr_strcat(file, value);

							FS.update_path(exclusive, _import_spawns_, file);

							//CALC PATH 2
							string256 file2 = { 0 };
							sprintf(file2, "scripts\\%s\\", path_name);
							xr_strcat(file2, value);

							FS.update_path(final_path, _game_config_, file2);

							Msg("PATH1: %s, Path2: %s", exclusive, final_path);

							FS.file_copy(exclusive, final_path);
						}
					}

					FS.file_copy(file.c_str(), create_file);
				}


			}

			if (sp)
			{
				sp->m_SpawnData.ModifyCustomData(spawnnew_custom_data);
			}

			i++;
		}
	}
}


