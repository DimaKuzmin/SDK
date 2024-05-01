#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "CustomObject.h"
#include <ESceneAIMapTools.h>

struct LevelSDK
{
	Fvector offset;
	xr_string name;
	xr_string path;
};

int cur_merge_levels = 0;
u32 loaded = 0;

xr_map<int, LevelSDK> level_offsets;
xr_map<int, xr_vector<CCustomObject*> > objects_loaded;


void UIObjectList::LoadFromMultiply()
{
	for (int i = 0; i < level_offsets.size(); i++)
	{
		LevelSDK l = level_offsets[i];
		ImportObjects(l.offset, true, l.path);
	}
}

void UIObjectList::ImportObjects(Fvector offset, bool use_path, xr_string path)
{
	xr_string temp_fn = "";

	loaded += 1;

	if (!use_path && EFS.GetOpenName(EDevice.m_hWnd, _import_, temp_fn) || use_path)
	{
		objects_to_move.clear();

		if (use_path)
			temp_fn = path;

		CInifile* file = xr_new<CInifile>(temp_fn.c_str(), true, true, true);

		string32 tmp; int i = 0;
		sprintf(tmp, "object_%d", i);

		if (!file)
		{
			Msg("Cant Read Inifile");
			return;
		}

		while (file->section_exist(tmp))
		{
			if (file->r_u32(tmp, "clsid") == (OBJCLASS_PORTAL | OBJCLASS_GROUP))
			{
				i++;
				sprintf(tmp, "object_%d", i);
				continue;
			}

			CCustomObject* obj = NULL;
			bool load = Scene->ReadObjectLTX(*file, tmp, obj);
			if (load)
			{
				Msg("Load: Sec: %s, Name: %s", tmp, obj->GetName());

				while (Scene->FindObjectByName(obj->GetName(), obj) != 0)
				{
					CCustomObject* obj_find = Scene->FindObjectByName(obj->GetName(), obj);
					xr_string name;
					string32 tmp;
					if (obj_find != nullptr)
					{
						name = obj_find->FName.c_str();
						name += "_";
						name += itoa(Random.randI(1, 20), tmp, 10);
						obj_find->SetName(name.c_str());
					}

				}

				if (!Scene->OnLoadAppendObject(obj))
					xr_delete(obj);

				objects_to_move[obj] = obj->GetPosition();

				Fvector3 pos = obj->GetPosition();
				pos.add(offset);
				obj->SetPosition(pos);


			}

			i++;
			sprintf(tmp, "object_%d", i);
		}
	}
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
			if (EFS.GetOpenPathName(EDevice.m_hWnd, _import_, buf_path, buf_name))
			{
				LevelSDK l;
				l.path = buf_path;
				l.name = buf_name;
				l.offset = { 0, 0, 0 };

				level_offsets[i] = l;
			}
		}

	}

	if (ImGui::Button("LoadFromOffsets", ImVec2(-1, 0)))
		LoadFromMultiply();
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

	objects_to_move.clear();
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


void UIObjectList::MoveObjectsToOffset()
{
	for (auto object : objects_to_move)
	{
		if (object.first != nullptr)
		{
			Fvector pos;
			pos.set(object.second);
			pos.add(vec_offset);
			object.first->SetPosition(pos);
		}
	}

	Scene->OnObjectsUpdate();
}