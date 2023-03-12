#include "stdafx.h"
#include "ui/UIObjectList.h"
#include "Edit\scene.h"
#include "Edit\ESceneCustomOTools.h"
#include "Edit\CustomObject.h"
#include "Edit\GroupObject.h"
#include "ESceneAIMapTools.h"
#include "ESceneSpawnTools.h"
#include "ESceneObjectTools.h"

#include "SceneObject.h"
#include "SpawnPoint.h"
#include "CustomObject.h"

struct LevelSDK
{
	Fvector offset;
	xr_string name;
	xr_string path;
};

xr_map<int, LevelSDK> level_offsets;
int size_merge_levels = 0;
int cur_merge_levels = 0;




xr_map<int, Fvector3> merge_offsets;	  
xr_map< CCustomObject*, Fvector3> objects_original_POS;

xr_map<u16, xr_vector<CCustomObject*> > objects_loaded;
u16 loaded = 0;
Fvector3 vec_offset = Fvector().set(0, 0, 0);
xr_string last_file;

string_path prefix_name;

string_path custom_data;

string_path logic_sec = "smart_terrain", path_sec = "level_", name_sec = "test_";

bool create_cfg_file = true;
bool select_file = false;



string4096 buffer_squads;	 
 
void UIObjectList::ExportSelectObjects()
{
	xr_string temp_fn = "";

	if (EFS.GetSaveName(_import_, temp_fn))
	{
		m_cur_cls = LTools->CurrentClassID();

		CInifile file(temp_fn.c_str(), false, false, false);

		int i = 0;
		for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
		{
			ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);

			if (it->first == m_cur_cls)
			{
				ObjectList& lst = ot->GetObjects();

				Msg("Object Counts: %d", ot->ObjCount());

				for (auto obj : lst)
				{
					if (obj->Selected())
					{
						string32 buffer = { 0 };
						sprintf(buffer, "object_%d", i);
						Scene->SaveObjectLTX(obj, buffer, file);
						++i;
					}
				}
			}
		}

		file.save_as(temp_fn.c_str());

	}
}

void UIObjectList::ExportAllObjects()
{
	if (Scene->m_LevelOp.m_FNLevelPath.size() == 0)
		return;
		  
	ESceneCustomOTool* scene_object = dynamic_cast<ESceneCustomOTool*>(Scene->GetOTool(OBJCLASS_SCENEOBJECT));

	if (scene_object)
	{
		ObjectList& list = scene_object->GetObjects();

		for (auto obj : list)
		{
			string512 name;
			sprintf(name, "%s_%s", Scene->m_LevelOp.m_FNLevelPath.c_str(), obj->GetName() );
			obj->SetName(name);
		}
	}

	string_path name;
	string128 name_str;
	xr_strcpy(name_str, "\\export_all_objects\\");
	xr_strcat(name_str, Scene->m_LevelOp.m_FNLevelPath.c_str());
	FS.update_path(name, _import_, name_str);

	//if (EFS.GetOpenName(EDevice.m_hWnd, _import_, temp_fn))
	{
		CInifile* file = xr_new<CInifile>(name, false, false, false);
		int i = 0;

		for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
		{
			ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);

			ObjectList& list = ot->GetObjects();
			if (ot)
			{
				if (ot->FClassID == OBJCLASS_DUMMY)
					continue;
 
				if (!ot->can_use_inifile())
					continue;

				for (auto obj : list)
				{
					if (!obj->Visible())
						continue;

					string64 tmp, str;
					xr_strcpy(str, "object_");
					xr_strcat(str, itoa(i, tmp, 10));
					Scene->SaveObjectLTX(obj, str, *file);//obj->SaveLTX(*file, str);
					i += 1;
					//Msg("ID: %d", i);
				}
			}
		}
		file->save_as(name);
	}

	ExportAIMap();

	/*
	if (Scene->GetTool(OBJCLASS_DO))
	{
		Msg("OBJCLASS_DO");

		string128 map = { 0 };
		xr_strcat(map, name);
		xr_strcat(map, ".detail");

		IWriter* write = FS.w_open_ex(map);
		Scene->GetTool(OBJCLASS_DO)->SaveStream(*write);
		FS.w_close(write);
		//Scene->GetTool(OBJCLASS_DO)->SaveStream();
	}
	*/
}

void UIObjectList::ExportAIMap()
{
	if (Scene->GetTool(OBJCLASS_AIMAP))
	if (Scene->GetTool(OBJCLASS_AIMAP)->Valid())
	{
		Msg("AI MAP");
 		string_path path;
		string128 name_str = {0};
		xr_strcpy(name_str, "\\export_all_objects\\");
		xr_strcat(name_str, Scene->m_LevelOp.m_FNLevelPath.c_str());
		FS.update_path(path, _import_, name_str);
 		xr_strcat(path, ".ai");

		IWriter* writer = FS.w_open_ex(path);
		ESceneAIMapTool* ai_tool = (ESceneAIMapTool*)Scene->GetTool(OBJCLASS_AIMAP);
		ai_tool->SaveStreamPOS(*writer);

		FS.w_close(writer);

	}
}

void UIObjectList::AddSceneObjectToList()
{
	ESceneCustomOTool* scene_objects = dynamic_cast<ESceneCustomOTool*>(Scene->GetOTool(OBJCLASS_SCENEOBJECT));
	ESceneAIMapTool* ai_map = dynamic_cast<ESceneAIMapTool*>(Scene->GetOTool(OBJCLASS_AIMAP));

	if (ai_map && scene_objects)
 		ai_map->SetSnapList(&scene_objects->GetObjects());
}


void UIObjectList::SetScale(Fvector size)
{
	if (m_SelectedObject)
	{
		m_SelectedObject->SetScale(size);
	}
}

void UIObjectList::ImportObjects(Fvector offset, bool use_path, xr_string path)
{
	xr_string temp_fn = "";

	loaded += 1;
	objects_original_POS.clear();
 
	if (!use_path && EFS.GetOpenName(EDevice.m_hWnd, _import_, temp_fn) || use_path)
	{
		if (use_path)
			temp_fn = path;

		CInifile file(temp_fn.c_str(), true, true, true);
		 



		string32 tmp; int i = 0;
		sprintf(tmp, "object_%d", i);

		u32 old_class = 0;

		while (file.section_exist(tmp))
		{
			//Msg("Read %s", tmp);

			if (file.r_u32(tmp, "clsid") == (OBJCLASS_PORTAL | OBJCLASS_GROUP))
			{
				i++;
				sprintf(tmp, "object_%d", i);
				continue;
			}

			CCustomObject* obj = NULL;
			bool load = Scene->ReadObjectLTX(file, tmp, obj);

			while (Scene->FindObjectByName(obj->GetName(), obj) != 0)
			{
				CCustomObject* obj_find = Scene->FindObjectByName(obj->GetName(), obj);
				xr_string name;
				string32 tmp;
				name = obj_find->FName.c_str();
				name += "_";
				name += itoa(Random.randI(1, 20), tmp, 10);
				obj_find->SetName(name.c_str());
				//Msg("Renamed %s", name.c_str());
			}

			if (load)
			{
				if (!Scene->OnLoadAppendObject(obj))
					xr_delete(obj);

				objects_original_POS[obj] = obj->GetPosition();

				Fvector3 pos = obj->GetPosition();
				pos.add(offset);
				obj->SetPosition(pos);
				//obj->FPosition.add(offset);
				if (old_class != obj->FClassID)
				{
					//Msg("Load [%d]", obj->FClassID);
					old_class = obj->FClassID;
				}
			}

			i++;
			sprintf(tmp, "object_%d", i);
			objects_loaded[loaded].push_back(obj);
		}
	}
   


	//Msg("Memory Use: %d", objects_loaded.get_allocator().max_size());
}

void UIObjectList::ImportMultiply()
{
	ImGui::InputInt("merge_multi: ", &cur_merge_levels, 1, 1);
  	for (int i = 0; i < level_offsets.size(); i++)
	{
		float value[3] = { level_offsets[i].offset.x, level_offsets[i].offset.y, level_offsets[i].offset.z };

 		if (ImGui::InputFloat3(level_offsets[i].name.c_str(), value, 0.0001f))
			level_offsets[i].offset.set(value);
	}

	if (ImGui::Button("LoadFiles", ImVec2(-1, 0)))
	{
		level_offsets.clear();

		xr_string buf_path = { 0 };
		xr_string buf_name = { 0 };


		for (int i = 0; i < cur_merge_levels; i++)
		{
			if (EFS.GetOpenPathName(EDevice.m_hWnd, _import_, buf_path, buf_name) )
			{
				//Msg("Load: %s", buf_path.c_str());
				//Msg("Name: %s", buf_name.c_str());
 
				LevelSDK l;
				l.path = buf_path;
				l.name = buf_name;
				l.offset = {0, 0, 0};

				level_offsets[i] = l;
			}
		}
			
	}
	
	if (ImGui::Button("LoadFromOffsets", ImVec2(-1, 0)))
	{
		LoadFromMultiply();
	}

	/*
	for (int i = 0; i != cur_merge_levels; i++)
	{
		float value[3] = { level_offsets[i].x, level_offsets[i].y, level_offsets[i].z };
		string32 offset = { 0 };
		sprintf(offset, "offset_%d", i);
		if (ImGui::InputFloat3(offset, value, 0.0001f))
			level_offsets[i].set(value);
	}
	*/
}

void UIObjectList::LoadFromMultiply()
{
	for (int i = 0; i < level_offsets.size(); i++)
	{
		LevelSDK l = level_offsets[i]; 
		ImportObjects(l.offset, true, l.path);
	}
}

void UIObjectList::SaveSelectedObjects()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	xr_string temp_fn = "";
	if (EFS.GetSaveName(_import_, temp_fn))
	{
		IWriter* write = FS.w_open_ex(temp_fn.c_str());

		for (auto obj : list)
		{
			CSceneObject* object_scene = smart_cast<CSceneObject*>(obj);

			if (obj->Selected() && object_scene)
			{
				write->open_chunk(EOBJ_CHUNK_OBJECT_BODY);
				CEditableObject* edit_obj = object_scene->GetReference();


				edit_obj->a_vPosition = obj->GetPosition();
				edit_obj->a_vRotate = obj->GetRotation();

				edit_obj->Save(*write);
				write->close_chunk();
			}

		}


		FS.w_close(write);
	}


}

void UIObjectList::CopyTempLODforObjects()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	for (auto obj : list)
	{
		CSceneObject* object_scene = (CSceneObject*)(obj);
		if (object_scene && object_scene->IsMUStatic())
		{
			CEditableObject* E = object_scene->GetReference();
			xr_string lod_name = E->GetLODTextureName();
			xr_string l_name = lod_name.c_str();

			string_path fn;
			int age, age_nm;

			FS.update_path(fn, _game_textures_, EFS.ChangeFileExt(l_name, ".dds").c_str());


			if (!FS.exist(fn))
			{
				string_path file;
				FS.update_path(file, _game_textures_, "lod_test\\lod_01.dds");
				if (FS.exist(file))
					FS.file_copy(file, fn);

				l_name += "_nm";

				FS.update_path(fn, _game_textures_, EFS.ChangeFileExt(l_name, ".dds").c_str());
				FS.update_path(file, _game_textures_, "lod_test\\lod_01_nm.dds");
				if (FS.exist(file))
					FS.file_copy(file, fn);
			}


		}
	}
}

bool UIObjectList::ExportDir(xr_string& dir)
{
	if (EFS.GetSaveName(_import_, dir))
		return true;

	return false;
}

bool UIObjectList::LoadAiMAP()
{
	if (last_file.size() == 0)
		return false;

	string128 temp = { 0 };
	xr_strcat(temp, last_file.c_str());

	if (Scene->GetTool(OBJCLASS_AIMAP))
	{
		//Msg("FIND AI MAP");
		IReader* read = FS.r_open(temp);
		Scene->GetTool(OBJCLASS_AIMAP)->LoadStreamOFFSET(*read, vec_offset);
		FS.r_close(read);
	}

	last_file.clear();

	return false;
}

void UIObjectList::UndoLoad()
{
	if (loaded > 0)
	{
		for (auto obj : objects_loaded[loaded])
		{
			obj->DeleteThis();
			Scene->RemoveObject(obj, false, true);
		}
		objects_loaded[loaded].clear_and_free();
		loaded -= 1;
	}
}

void UIObjectList::SelectLoaded()
{
	if (loaded > 0)
	{
		for (auto obj : objects_loaded[loaded])
		{
			obj->Select(true);
		}
	}
}

void UIObjectList::CheckCustomData()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SPAWNPOINT));

	ObjectList& list = ot->GetObjects();
	for (auto item : list)
	{
		CSpawnPoint* sp = (CSpawnPoint*)item;
		if (sp)
		{
			shared_str name = sp->m_SpawnData.ReadCustomData();
			if (name.size() > 0)
				Msg("CUSTOM: %s", name.c_str());
		}

	}
}

void UIObjectList::MoveObjectsToOffset()
{
	for (auto obj : objects_original_POS)
	{
		Fvector pos;
		pos.set(obj.second);
		pos.add(vec_offset);
		obj.first->SetPosition(pos);
	}

	Scene->OnObjectsUpdate();
}

void UIObjectList::CheckDuplicateNames()
{
	 
	xr_vector<LPCSTR> names;

	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

	ObjectList& list = ot->GetObjects();

	for (auto item : list)
	{
		if (Scene->FindObjectByName(item->GetName(), item) != 0)
		{
			string256 name = { 0 };
			xr_strcat(name, item->GetName());
			xr_strcat(name, "_");
			string32 tmp;

			while (Scene->FindObjectByName(item->GetName(), item) != 0)
			{
				xr_strcat(name, itoa(Random.randI(1, 20), tmp, 10));
				Msg("Rename Obj %s", name);
				item->SetName(name);
			}
		}
	}

	names.clear_and_free();

	/*
	xr_map<LPCSTR, u16> map_list;

	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
	   

		ObjectList& list = ot->GetObjects();
		if (ot)
		{
			for (auto item : list)
			{
 				map_list[item->RefName()] += 1;
				int id = map_list[item->RefName()];
				
				string256 name = {0}, tmp;

				xr_strcat(name, item->RefName());
				xr_strcat(name, "_");
				xr_strcat(name, itoa(id, tmp, 10));

				item->SetName(name);
			}
		}


	}

	map_list.clear();
	*/
}

bool sort_list(CCustomObject* obj1, CCustomObject* obj2)
{
	if (obj1->RefName() && obj2->RefName())
	if (xr_strcmp(obj1->RefName(), obj2->RefName() ) < 0) 
		return true;

	return false;
};

void UIObjectList::RenameALLObjectsToObject()
{	
	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		
		if (!ot)
			continue;

		if (ot->FClassID == OBJCLASS_LIGHT ||
			ot->FClassID == OBJCLASS_GLOW || 
			ot->FClassID == OBJCLASS_SECTOR || 
			ot->FClassID == OBJCLASS_PORTAL || 
			ot->FClassID == OBJCLASS_PS
		)
		{
			ObjectList list = ot->GetObjects();
			string256 name_prefix = { 0 }, tmp;
			xr_string tool_class ;
			
			if (ot->FClassID == OBJCLASS_LIGHT)
				tool_class = "light";
			else if (ot->FClassID == OBJCLASS_GLOW)
				tool_class = "glow";
			else if (ot->FClassID == OBJCLASS_SECTOR)
				tool_class = "sector";
			else if (ot->FClassID == OBJCLASS_PORTAL)
				tool_class = "portal";
			else if (ot->FClassID == OBJCLASS_PS)
				tool_class = "ps";

			int id = 1;

			for (auto item : list)
			{
				string256 name_prefix = { 0 }, tmp;
				xr_strcat(name_prefix, tool_class.c_str());
				xr_strcat(name_prefix, "_");
				xr_strcat(name_prefix, itoa(id, tmp, 10));

				item->SetName(name_prefix);
				id++;
			}			 
		}


		if (ot->FClassID == OBJCLASS_SCENEOBJECT )  // ot->FClassID == OBJCLASS_SPAWNPOINT 
		{
			int id = 1;

			xr_map<LPCSTR, u16> map_names_ref;

			ObjectList list = ot->GetObjects();
			for (auto item : list)
			{
			//	Msg("Name %s", item->RefName());

				string256 prefix = { 0 };
				if ( item->RefName() )
				{
					map_names_ref[item->RefName()] += 1;
					id = map_names_ref[item->RefName()];
					xr_strcat(prefix, item->RefName());
				}
				else
				{
					xr_strcat(prefix, ot->FClassID == OBJCLASS_SPAWNPOINT ? "spawn_no_ref" : "static_no_ref");
					id++;
				}
				
				

				string256 name_new = {0}, tmp;
				xr_strcat(name_new, prefix);
				xr_strcat(name_new, "_");
				xr_strcat(name_new, itoa(id, tmp, 10) );

				item->SetName(name_new);
 			}
 
			list.sort(sort_list);
		}
		
 
	}
}

void UIObjectList::BboxSelectedObject()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

	ObjectList& list = ot->GetObjects();

	Fbox box_all;
	for (auto item : list)
	{
		if (!item->Selected())
			continue;

		Fbox box;
		item->GetBox(box);

		box_all.merge(box);
		Msg("BBOX x[%f][%f]", box.x1, box.x2);
		Msg("BBOX z[%f][%f]", box.z1, box.z2);
		Msg("BBOX y[%f][%f]", box.y1, box.y2);
	}

	Msg("Selected BBOX x[%f][%f]", box_all.x1, box_all.x2);
	Msg("Selected BBOX z[%f][%f]", box_all.z1, box_all.z2);
	Msg("Selected BBOX y[%f][%f]", box_all.y1, box_all.y2);

}

void UIObjectList::POS_ObjectsToLTX()
{
	xr_string file;
	if (EFS.GetSaveName(_import_, file))
	{
		CInifile* ini_file = xr_new<CInifile>(file.c_str(), false, false, false);

		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
		
		ObjectList& list = ot->GetObjects();

		for (auto item : list)
		{
			if (!item->Selected())
				continue;

			ini_file->w_fvector3(item->GetName(), "position", item->GetPosition());
		}

		ini_file->save_as(file.c_str());
	}
}

void UIObjectList::SetTerrainOffsetForAI()
{
	ESceneToolBase* tool = Scene->GetTool(OBJCLASS_AIMAP);

	ObjectList* list = tool->GetSnapList();
	for (auto obj : *list)
	{
		if (obj->Selected())
			vec_offset = obj->GetPosition();
	}
}

void UIObjectList::SelectAIMAPFile()
{
	xr_string file;
	if (EFS.GetOpenName(EDevice.m_hWnd, _import_, file))
	{
		last_file = file;
	}
}

void UIObjectList::ModifyAIMAPFiles(Fvector offset)
{
	xr_string file;
	if (EFS.GetOpenName(EDevice.m_hWnd, _import_, file))
	{

		IReader* read = FS.r_open(file.c_str());
		
		xr_vector<Fvector3> positions;

		if (read)
		{

			read->open_chunk(2);
			
			u32 size = read->r_u32();
			
			for (int i = 0; i < size; i++)
			{
				Fvector3 pos;
				pos.x = read->r_float();
				pos.y = read->r_float();
				pos.z = read->r_float();
				pos.add(offset);
				positions.push_back(pos);
			}
			
		}

		FS.r_close(read);

		string128 file_end = {0};
		xr_strcat(file_end, file.c_str());
		xr_strcat(file_end, ".new");


		IWriter* write = FS.w_open(file_end);

		if (write)
		{
			write->open_chunk(2);
			write->w_u32(positions.size());

			for (auto pos : positions)
			{
				write->w_float(pos.x);
				write->w_float(pos.y);
				write->w_float(pos.z);
			}  
			write->close_chunk();
		}

		FS.w_close(write);
	}
}

int value_ai;


xr_vector<Fvector3> UIObjectList::getAIPOS(LPCSTR file)
{
	IReader* read = FS.r_open(file);

	xr_vector<Fvector3> positions;

	if (read)
	{

		read->open_chunk(2);

		u32 size = read->r_u32();

		for (int i = 0; i < size; i++)
		{
			Fvector3 pos;
			read->r_fvector3(pos);
			positions.push_back(pos);
		}

	}

	FS.r_close(read);

	return positions;
}
 
void UIObjectList::MergeAIMAP(u32 files)
{
	xr_vector <Fvector3> result;

	for (int i = 0; i < files; i++)
	{
		xr_string file;
		if (EFS.GetOpenName(EDevice.m_hWnd, _import_, file))
		{	   
			for (auto pos : getAIPOS(file.c_str()))
			{
				pos.add(merge_offsets[i]);
				result.push_back(pos);
			}
		}
	}


	xr_string save_file;
	if (EFS.GetSaveName(_import_, save_file))
	{
		IWriter* write = FS.w_open(save_file.c_str());
		write->open_chunk(2);
		write->w_u32(result.size());
		for (auto pos : result)
		{
			write->w_fvector3(pos);
		}
		write->close_chunk();
		FS.w_close(write);
	}

}

void UIObjectList::RenameSelectedObjects()
{
	ESceneCustomOTool*  base = Scene->GetOTool(LTools->CurrentClassID());
	int i = 0;
	for (auto item : base->GetObjects())
	{
		if (item->Selected())
		{
			string256 name;

			sprintf(name, "%s_%d", &prefix_name, i);

			item->SetName(name);

			i++;
		}
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
			CSpawnPoint* sp = (CSpawnPoint*) item;
			
			if (sp && create_cfg_file || sp && select_file)
			{	 
				string4096 text = {0};
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
					
					if (50 < Random.randI(0, 100) )
						monster = true;
 

					if (!monster)
						file->w_string("spawn_sim", "spawn_squads", squads[Random.randI(1, 9)] );
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
							string256 file = {0};
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
				sp->m_SpawnData.ModifyCustomData(custom_data);
			}

			i++;
		}
	}
}

void UIObjectList::CreateLogicConfigs()
{
	xr_string file_path;
	string_path ex = {0};

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
			size_anims   = read_params->r_u32("animpoint", "anims");
			
			include_sect_name = read_params->r_string("animpoint", "include_sect");
			meet_sect_name	 = read_params->r_string("animpoint", "meet_sect");

			sect = read_params->r_section(include_sect_name);
			meet_sect = read_params->r_section(meet_sect_name);



			smart_file = read_params->r_string("animpoint", "logic_cfg");
			smart_prefix = read_params->r_string("animpoint", "logic_prefix");

			

			u32 items = _GetItemCount(camps);
			for (int i = 0; i < items;i++)
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
		 
			file->w_string(logic,   "active", active);
			file->w_u32(logic,      "prior", 100);
			file->w_string(logic,   "suitable", "true");
			
			string32 cover_name;
			sprintf(cover_name, "%s_anim_%d", smart_prefix, i);
			
			file->w_string(active, "cover_name", cover_name);
			file->w_string(active, "use_camp", camps_ids[i] ? "true" : "false");
			file->w_section_include(active, sect.Name.c_str());
		}

		xr_strcat(ex, file_path.c_str());
		xr_strcat(ex, ".logic");
		file->save_as(ex);

		string_path smart_path = {0};
		xr_strcat(smart_path, file_path.c_str());
		xr_strcat(smart_path, ".smart");
		CInifile* ini_smart = new CInifile(smart_path, false, false);
		
		for (int i = 0; i < size_anims;i++)
		{
			string32 logic;
			sprintf(logic, "anim_%d", i);

			ini_smart->w_string("exclusive", logic, smart_file);
		}
		ini_smart->save_as(smart_path);
	}
}

int sector = 0;
int cur_light_type = 0;
const char* light_type[] = { "all", "lightmap", "dynamic", "animated"};

int cur_smart = 0;

void UIObjectList::UpdateUIObjectList()
{

 	{
		ImGui::Text("OFFSET: ");
		float vec[3] = { vec_offset.x, vec_offset.y, vec_offset.z };

		if (ImGui::InputFloat3("x", vec, 0.001f))
			vec_offset.set(vec[0], vec[1], vec[2]);
	}

	if (LTools->CurrentClassID() == OBJCLASS_SECTOR)
	{
		ImGui::InputInt("sector_id", &sector, 1, 100);
		if (ImGui::Button("FIND SECTOR BY ID"))
			FindObjectSector(sector);
	}

	if (LTools->CurrentClassID() == OBJCLASS_LIGHT)
		ImGui::ListBox("list_light", &cur_light_type, light_type, IM_ARRAYSIZE(light_type), 8);

	if (LTools->CurrentClassID() != OBJCLASS_AIMAP && LTools->CurrentClassID() != OBJCLASS_DO && LTools->CurrentClassID() != OBJCLASS_GROUP)
	{
		if (ImGui::Button("Move Loaded to Offset", ImVec2(-1, 0)))
			MoveObjectsToOffset();

		if (ImGui::Button("Export Level ALL", ImVec2(-1, 0)))
			ExportAllObjects();

		if (ImGui::Button("Load Objects", ImVec2(-1, 0)))
			ImportObjects(vec_offset);

		if (ImGui::Button("Export Objects", ImVec2(-1, 0)))
			ExportSelectObjects();

		if (ImGui::Button("Undo load", ImVec2(-1, 0)))
			UndoLoad();

		if (ImGui::Button("sel loaded", ImVec2(-1, 0)))
			SelectLoaded();

		if (ImGui::Button("check duplicate", ImVec2(-1, 0)))
			CheckDuplicateNames();

		if (ImGui::Button("rename all objects", ImVec2(-1, 0) ) )
			RenameALLObjectsToObject();

	}
	
	if (LTools->CurrentClassID() == OBJCLASS_AIMAP)
	{
		/*
		ImGui::InputInt("SelectID: ", &value_ai, 0, 100000000);

		if (ImGui::Button("SelectNode", ImVec2(-1, 0)))
		{
			ESceneAIMapTool* ai_tool = (ESceneAIMapTool*)Scene->GetTool(OBJCLASS_AIMAP);
			if (ai_tool)
				ai_tool->SelectNode(value_ai);
		}
		
		if (ImGui::Button("Set SnapList", ImVec2(-1, 0)))
			AddSceneObjectToList();
		*/

		if (ImGui::Button("select map file", ImVec2(-1, 0)))
			SelectAIMAPFile();

		if (ImGui::Button("load map", ImVec2(-1, 0)))
			LoadAiMAP();
		if (ImGui::Button("export map", ImVec2(-1, 0)))
			ExportAIMap();

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
	}

	if (LTools->CurrentClassID() == OBJCLASS_SCENEOBJECT)
	{
		if (ImGui::Button("temp lods", ImVec2(-1, 0)))
			CopyTempLODforObjects();

		if (ImGui::Button("save object", ImVec2(-1, 0)))
			SaveSelectedObjects();

		if (ImGui::Button("bbox object", ImVec2(-1, 0)))
			BboxSelectedObject();

		if (ImGui::Button("pos_objects_save_ltx", ImVec2(-1, 0) ))
			POS_ObjectsToLTX();

		ImportMultiply();
	}

	if (LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT)
	{
		//if (ImGui::Button("Check CustomData", ImVec2(-1, 0)))
		//	CheckCustomData();

		ImGui::InputText("#repace_name", prefix_name, sizeof(prefix_name));

		if (ImGui::Button("rename_selected", ImVec2(-1, 0)))
		{
			RenameSelectedObjects();
		}

		ImGui::InputInt("smart_mask", &cur_smart, 1, 10);

		if (ImGui::Button("smart_terrain_mask export", ImVec2(-1, 0)))
		{
			ESceneCustomOTool* scene_object = dynamic_cast<ESceneCustomOTool*>(Scene->GetOTool(OBJCLASS_SPAWNPOINT));
			ObjectList list = scene_object->GetObjects();

			for (auto obj : list)
			{
			   if (obj->Selected())
			   {
				   Msg("[%s]", obj->GetName());
				   Msg("  255, 255, 255, %d", cur_smart);
						  
			   }
			}
		}	

		if (ImGui::Button("smart_terrain_names", ImVec2(-1, 0)))
		{
			ESceneCustomOTool* scene_object = dynamic_cast<ESceneCustomOTool*>(Scene->GetOTool(OBJCLASS_SPAWNPOINT));
			ObjectList list = scene_object->GetObjects();

			for (auto obj : list)
			{
				if (obj->Selected())
					Msg("%s", obj->GetName());
			}
		}

		/*

		if (create_cfg_file)
		{
			ImGui::InputText("#logic_sec", logic_sec, sizeof(logic_sec));
			ImGui::InputText("#path_sec", path_sec, sizeof(path_sec));
		}
		else
		{
			if (!select_file)
			{
				ImGui::InputText("#custom_data", custom_data, sizeof(custom_data));
			}
		}

	  
		ImGui::Checkbox("#create_file", &create_cfg_file);
		ImGui::Checkbox("#select_file", &select_file);

		if (ImGui::Button("#custom_logic", ImVec2(-1, 0)) )
		{
			SetCustomData(true, logic_sec, path_sec);
		}
 
		if (ImGui::Button("#create_logic", ImVec2(-1, 0)))
		{
			CreateLogicConfigs();
		}

		*/
	}

}
 
int current_selected_item = 0;
const char* items[] = { "null", "use_items", "smart_terrain", "space_restrictor", "smart_cover", "camp_zone", "zones", "campfire", "graph_point", "anomal_zone" };
const char* fireboll[] = { "fireboll", "fireboll_acidic", "fireboll_electric" };
const char* fild_zones[] = { "field_acidic", "field_psychic", "field_radioactive", "field_thermal" };
const char* mine_zones[] = { "mine_acidic", "mine_electric", "mine_gravitational", "mine_thermal" };
const char* use_items[] = { "antirad", "bandage", "conserva", "drug_", "energy_drink", "harmonica_a", "guitar_a", "kolbase", "vodka", "medkit", "wpn_", "outfit" };

#include "ELight.h"

bool UIObjectList::CheckNameForType(CCustomObject* obj)
{
	if (LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT)
	{
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

					if (m_Filter_type[0])
					{
						const char* str = strstr(obj->RefName(), m_Filter_type);
						if (!str)
							return false;

						//	Msg("size %d", strlen(str) );
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
	
	if (LTools->CurrentClassID() == OBJCLASS_LIGHT)
	{
		CLight* light = (CLight*)obj;
		if (light)
		{
			bool light_map = light->m_Flags.is(ELight::flAffectStatic);
			bool dynamic   = light->m_Flags.is(ELight::flAffectDynamic);
			bool animated = light->m_Flags.is(ELight::flProcedural);
			if (light_map && cur_light_type == 1)
				return true;

			if (dynamic && cur_light_type == 2)
				return true;
			
			if (animated && cur_light_type == 3)
				return true;

			if (cur_light_type == 0)
				return true;

			return false;
		}
	}


	return true;
}

void UIObjectList::ListBoxForTypes()
{
	if (current_selected_item == 0)
	{
		ImGui::Text("GameType Name:");
		//ImGui::SetNextItemWidth(-200 );
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


