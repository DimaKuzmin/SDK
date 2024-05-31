#include "stdafx.h"
#include "ui/UIObjectList.h"
#include "Edit\scene.h"
#include "Edit\ESceneCustomOTools.h"
#include "Edit\CustomObject.h"
#include "Edit\GroupObject.h"
#include "ESceneAIMapTools.h"
#include "ESceneSpawnTools.h"
#include "ESceneObjectTools.h"
#include "ESceneWayTools.h"

#include "SceneObject.h"
#include "SpawnPoint.h"
#include "CustomObject.h"
#include "WayPoint.h"
 
#include "../xrServerEntities/xrServer_Objects_Alife_Smartcovers.h"
#include "SpawnPoint.h"
 
void UIObjectList::UpdateDefaultMeny()
{
	if (LTools->CurrentClassID() == OBJCLASS_AIMAP)
	{
		ImGui::Text("AIMAP: ");

		ImGui::Text("OFFSET: ");
		float vec[3] = { vec_offset.x, vec_offset.y, vec_offset.z };

		if (ImGui::InputFloat3("x", vec, 0.001f))
			vec_offset.set(vec[0], vec[1], vec[2]);

		ImGui::Checkbox("ignore_construction_on_load", &ai_ignore_stractures);

		if (ImGui::Button("select map file", ImVec2(-1, 0)))
			SelectAIMAPFile();

		if (ImGui::Button("load map", ImVec2(-1, 0)))
		{
			ESceneToolBase* tool = Scene->GetTool(OBJCLASS_AIMAP);
			ESceneAIMapTool* tool_ai = smart_cast<ESceneAIMapTool*>(tool);
			tool_ai->CreateCFModel();
			LoadAiMAP();
		}

		if (ImGui::Button("export map", ImVec2(-1, 0)))
			ExportAIMap(0, Scene->m_LevelOp.m_FNLevelPath.c_str());

		/*
		if (ImGui::Button("set offset terrain", ImVec2(-1, 0)))
			SetTerrainOffsetForAI();

		if (ImGui::Button("move to offsets", ImVec2(-1, 0)))
			ModifyAIMAPFiles(vec_offset);			
		

		
		if (ImGui::Button("merge", ImVec2(-1, 0)))
			MergeAIMAP(merge_ai_map_size);

		ImGui::InputInt("size: ", &merge_ai_map_size, 1, 1);
		if (merge_ai_map_size > 0)
		{
			for (int i = 0; i < merge_ai_map_size; i++)
			{
				float value[3] = { merge_offsets[i].x, merge_offsets[i].y, merge_offsets[i].z };
				string32 offset = { 0 };
				string32 tmp;
				xr_strcat(offset, "ofsset_");
				xr_strcat(offset, itoa(i, tmp, 10));
				if (ImGui::InputFloat3(offset, value, 0.0001f))
					merge_offsets[i].set(value);

			}
		}

		*/

		if (ImGui::Button("merge_ini", ImVec2(-1, 0)))
		{
			xr_string path;
			if (EFS.GetOpenName(EDevice.m_hWnd, _import_, path))
			{
				CInifile* file = new CInifile(path.c_str(), true);
				MergeAI_FromINI(file);
			}
		}

		ImGui::Separator();
	}

	if (LTools->CurrentClassID() == OBJCLASS_SCENEOBJECT)
	{
		ImGui::Text("SCENE: ");
		

		if (ImGui::Button("temp lods", ImVec2(-1, 0)))
			CopyTempLODforObjects();

		ImGui::Checkbox("use_global_pos", &use_global_position);
		if (ImGui::Button("save object", ImVec2(-1, 0)))
			SaveSelectedObjects();

		if (ImGui::Button("pos_objects_save_ltx", ImVec2(-1, 0) ))
			POS_ObjectsToLTX();

		ImportMultiply();

		ImGui::Separator();
	}

	if (LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT)
	{
		ImGui::Text("SPAWN: ");
		if (ImGui::Button("replace_to_phobject", ImVec2(-1, 0)))
			ReplaceItemToPHYSIC_STATIC();

		if (ImGui::Button("Update_ReplaceNames"))
			UpdateReplaceNames();
		
		ImGui::Checkbox("GenConfigs", &use_genarate_cfgs);
		ImGui::Checkbox("IgnoreVisual", &IgnoreVisual);
		ImGui::Checkbox("IgnoreNotVisual", &IgnoreNotVisual);
		ImGui::Checkbox("IgnoreCombatCover", &IgnoreCombatCovers);

		if (ImGui::Checkbox("Show Only CustomData Objects", &current_only_customdata))
			UpdateCustomData();

		if (ImGui::Button("Error_Spawns", ImVec2(-1, 0)))
			LoadErrorsGraphs();

		if (ImGui::Button("Hide_CombatCovers", ImVec2(-1, 0)))
			HideCombatCovers();

		if (ImGui::Button("Game Graphs(ClearLevels)"))
			ClearGraphs();

		if (use_genarate_cfgs)
		{
			ImGui::InputText("section", prefix_cfg_section, sizeof(prefix_cfg_section));
			ImGui::InputText("map", prefix_cfg_map, sizeof(prefix_cfg_map));
			ImGui::InputText("prefix", prefix_cfg_prefix, sizeof(prefix_cfg_prefix));


			if (ImGui::Button("GenSpawnCFG", ImVec2(-1, 0)))
				GenSpawnCFG(prefix_cfg_section, prefix_cfg_map, prefix_cfg_prefix);

			if (ImGui::Button("Check CustomData", ImVec2(-1, 0)))
				CheckCustomData();

			if (ImGui::Button("Clear_CustomData", ImVec2(-1, 0)))
				ClearCustomData();
		}
		
		ImGui::Separator();

	}
}



void UIObjectList::UpdateUIObjectList()
{
 	{
		ImGui::Text("New Menu: ");
		ImGui::Checkbox("LoadMenu", &ShowLOAD);
		ImGui::Checkbox("ExportMenu", &ShowEXPORT);
		ImGui::Checkbox("Rename_CheckForErrors", &ShowRenamer);

		ImGui::Separator();

		ImGui::Text("Feature: ");
		ImGui::Checkbox("MultiplySelect", &MultiplySelect);
		ImGui::Checkbox("ShowError", &use_errored);
		ImGui::Checkbox("Use Distances", &use_distance);
		if (use_distance)
			ImGui::InputInt("Distance", &DistanceObjects, 1, 100);

		ImGui::Separator();
	}

	UpdateDefaultMeny();
		
	if (LTools->CurrentClassID() != OBJCLASS_AIMAP && LTools->CurrentClassID() != OBJCLASS_DO && LTools->CurrentClassID() != OBJCLASS_GROUP)
	{
		if (ShowRenamer)
		{		
			ImGui::Text("Rename Menu: (TOOL CURRENT)");
			ImGui::Checkbox("use_prefix_by_refname", &use_prefix_refname);

			if (use_prefix_refname)
				ImGui::InputText("#repace_name (prefix)", rename_prefix_name, sizeof(rename_prefix_name));

			if (ImGui::Button("rename (selected)", ImVec2(-1, 0)))
				RenameSelectedObjects();
 
			if (ImGui::Button("Rename All (group#_No)", ImVec2(-1, 0) ) )
				RenameALLObjectsToObject();

			if (ImGui::Button("To Log All (only dup)", ImVec2(-1, 0)))
				FindALL_Duplicate();

			if (ImGui::Button("Rename All (only dup)", ImVec2(-1, 0)))
				CheckDuplicateNames();

			if (ImGui::Button("Names to LOG (FOR LTX)", ImVec2(-1, 0)))
			{
				ESceneCustomOTool* scene_object = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
				ObjectList list = scene_object->GetObjects();

				for (auto obj : list)
				{
					if (obj->Selected())
						Msg("%s", obj->GetName());
				}
			}

			ImGui::InputInt("MaxName (In Search)", &_sizetext, 1, 10);
			ImGui::Separator();
 		}
 
		if (ShowLOAD)
		{
			ImGui::Text("Load Menu:");

			ImGui::Text("OFFSET: ");
			float vec[3] = { vec_offset.x, vec_offset.y, vec_offset.z };

			if (ImGui::InputFloat3("x", vec, 0.001f))
				vec_offset.set(vec[0], vec[1], vec[2]);

			if (ImGui::Button("Move Loaded to Offset", ImVec2(-1, 0)))
  				MoveObjectsToOffset();
	 
			if (ImGui::Button("Load Objects", ImVec2(-1, 0)))
				ImportObjects(vec_offset);

			if (ImGui::Button("Undo load", ImVec2(-1, 0)))
				UndoLoad();
 
			if (ImGui::Button("Select Loaded", ImVec2(-1, 0)))
				SelectLoaded();

			ImGui::Separator();

		}	  

		if (ShowEXPORT)
		{
			ImGui::Text("Export Menu: ");

			if (ImGui::Button("Export Objects (Selected)", ImVec2(-1, 0)))
				ExportSelectObjects();
			if (ImGui::Button("Export Level ALL", ImVec2(-1, 0)))
				ExportAllObjects();

			float vec_min[3] = { vec_box_min.x, vec_box_min.y, vec_box_min.z };
			float vec_max[3] = { vec_box_max.x, vec_box_max.y, vec_box_max.z };

			if (ImGui::InputFloat3("box_min", vec_min, 0.001f))
				vec_box_min = {vec_min[0], vec_min[1], vec_min[2]};

			if (ImGui::InputFloat3("box_max", vec_max, 0.001f))
				vec_box_max = {vec_max[0], vec_max[1], vec_max[2]};

			if (ImGui::Button("bbox object", ImVec2(-1, 0)))
				BboxSelectedObject();
 
			ImGui::Checkbox("use_outside (EXPORT_IN_BOX)", &use_outside_box);

			if (ImGui::Button("Export Objects InBox" , ImVec2(-1, 0)  ) )
				ExportInsideBox();
			
			if (ImGui::Button("Remove Objects InBox", ImVec2(-1, 0)))
				RemoveAllInsideBox();

			if (ImGui::Button("Set ListMove From InBox", ImVec2(-1, 0)))
				SetListToMove();


			ImGui::Separator();
		}	 
	}
 

	if (LTools->CurrentClassID() == OBJCLASS_WAY)
	{
		if (ImGui::Button("check ways"))
		{
			ESceneWayTool* way_tool = (ESceneWayTool*)Scene->GetTool(OBJCLASS_WAY);
		
			if (way_tool)
			{
				auto list = way_tool->GetObjects();
				
				for (auto way : list)
				{
					int id = 0;
					CWayObject* way_object = smart_cast<CWayObject*>(way);
					
					if (way_object && way_object->Selected())
					{
	  					Msg("Test Way: %s", way->GetName());

						for (auto wayp : way_object->GetWPoints())
						{
							if (wayp)
							{  
 								Msg("Way[%u] Flag Set1: %u", id, wayp->m_Flags);
								//Msg("Flag Set2: %d", wayp->m_Flags.test(1));
								//Msg("Flag Set3: %d", wayp->m_Flags.test(2));
								//Msg("Flag Set4: %d", wayp->m_Flags.test(3));
								//Msg("Flag Set5: %d", wayp->m_Flags.test(4));
								//Msg("Flag Set6: %d", wayp->m_Flags.test(5));
								//Msg("Flag Set7: %d", wayp->m_Flags.test(6));
								//Msg("Flag Set8: %d", wayp->m_Flags.test(7));
							}
							id++;
						}
					}
				}

			}

		}
 	}

}
 
int current_selected_item = 0;
const char* items[] = { "null", "use_items", "smart_terrain", "space_restrictor", "smart_cover", "camp_zone", "zones", "campfire", "graph_point", "anomal_zone", "physic", "physic_object", "physic_d"};
const char* fireboll[] = { "fireboll", "fireboll_acidic", "fireboll_electric" };
const char* fild_zones[] = { "field_acidic", "field_psychic", "field_radioactive", "field_thermal" };
const char* mine_zones[] = { "mine_acidic", "mine_electric", "mine_gravitational", "mine_thermal" };
const char* use_items[] = { "antirad", "bandage", "conserva", "drug_", "energy_drink", "harmonica_a", "guitar_a", "kolbase", "vodka", "medkit", "wpn_", "outfit" };

#include "ELight.h"

xr_vector<CCustomObject*> objects_failed;
u32 OldDeviceTime = 0;



bool UIObjectList::CheckForError(CCustomObject* object)
{
	if (use_errored)
	{

		if (OldDeviceTime < EDevice.dwTimeGlobal)
		{
			OldDeviceTime = EDevice.dwTimeGlobal + 2000;
			
			for (auto item : Errored_objects)
			{
				if (object->FName.equal(item.c_str()))
				{
					objects_failed.push_back(object);
				}
			};
			
		}
 
		bool finditem = false;
		
		for (auto item : objects_failed)
		{
			if (item == object)
			{
				finditem = true;
				return false;
			}
		}
 
		if (finditem)
			return true;
	}
	else
	{
		if (object->FName.size() <= _sizetext)
			return true;
	}


	return false;
}

void UIObjectList::FindALL_Duplicate()
{
	Errored_objects.clear();
						
	ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	if (mt)
	{
		ObjectList& lst = mt->GetObjects();
		for (auto object : lst)
		{
			if (Scene->FindObjectByName(object->GetName(), object))
			{
				Msg("Duplicate object name already exists: '%s', Class: %d, Ref: %s, POS[%f][%f][%f]", object->GetName(), (mt->FClassID), object->RefName(), VPUSH(object->GetPosition()));
				Errored_objects.push_back(object->FName);
			}
		}
	}
}

void UIObjectList::LoadErrorsGraphs()
{
 	Errored_objects.clear();
	xr_string file_path;

	if (EFS.GetOpenName(EDevice.m_hWnd, _import_, file_path))
	{
		CInifile* file = xr_new<CInifile>(file_path.c_str());

		string128 name_line;
 		sprintf(name_line, "graph_%d", 0);
 
		int id = 0;
		while (file->line_exist("graphs", name_line))
		{
			id++;
			Errored_objects.push_back(file->r_string("graphs", name_line));
			sprintf(name_line, "graph_%d", id);
		}
		

	}
}
 
bool UIObjectList::CheckNameForType(CCustomObject* obj)
{
	if (!CheckForError(obj))
		return false;
 
	if (LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT)
	{
		if (IgnoreCombatCovers)
		{
			CSpawnPoint* spawn = smart_cast<CSpawnPoint*>(obj);
			if (spawn)
			{
				auto ise = spawn->m_SpawnData.m_Data;

				CSE_SmartCover* smart_cover = smart_cast<CSE_SmartCover*>(ise);
				if (smart_cover && smart_cover->m_is_combat_cover)
 					return false;
 			}
		}

		if (current_only_customdata)
		{
			auto Out = std::find(customdata_objects.begin(), customdata_objects.end(), obj);
			if (Out == customdata_objects.end())
				return false;
		}

		if (!current_selected_item && xr_strlen(m_Filter_type) == 0)
			return true;

		if (!obj->RefName())
			return false;

	
		if (current_selected_item != 0)
		{
			if (current_selected_item == 1)
			{
				bool find = false;
				for (auto item : use_items)
				{
					if (strstr(obj->RefName(), item))
						find = true;
				}

				if (!find)
					return false;
			}
			else
				if (current_selected_item == 6)
				{
					bool find = false;
					for (auto item : fireboll)
						if (strstr(obj->RefName(), item))
							find = true;

					for (int i = 0; i < 4; i++)
						if (strstr(obj->RefName(), fild_zones[i]) || strstr(obj->RefName(), mine_zones[i]))
							find = true;

					if (!find)
						return false;
				}
				else
				{
 					xr_strcpy(m_Filter_type, items[current_selected_item]);

					if (m_Filter_type[0] && obj->RefName() != 0)
					{
						const char* str = strstr(obj->RefName(), m_Filter_type);
						if (!str)
							return false;
					}
				}
		}
		else
		{
			if (m_Filter_type[0] && obj->RefName() != 0)
			if (strstr(obj->RefName(), m_Filter_type) == 0)
				return false;
		}
	}

	return true;
}

void UIObjectList::ListBoxForTypes()
{
	if (LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT)
	{
		ImGui::Text("VisualName:");
		ImGui::InputText("##value_visual", m_Filter_visual, sizeof(m_Filter_visual));
	}

	if (current_selected_item == 0)
	{
		ImGui::Text("GameType Name:");
 		ImGui::InputText("##value_type", m_Filter_type, sizeof(m_Filter_type));
	}

	ImGui::Text("Select Type:");
	ImGui::ListBox("", &current_selected_item, items, IM_ARRAYSIZE(items), 8);
}

void UIObjectList::FindObjectSector(u16 id)
{
	ESceneCustomOTool* otool = Scene->GetOTool(OBJCLASS_SECTOR);
	if (otool)
	{
		int i = 0;
		for (auto sector : otool->GetObjects())
		{
			if (i == id)
			{
				Msg("SECTOR NAME: %s", sector->GetName());
				sector->Select(true);
			}
			
			i++;
		}
	}

	
}


