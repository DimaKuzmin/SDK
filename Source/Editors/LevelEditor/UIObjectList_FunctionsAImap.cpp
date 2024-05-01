#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "CustomObject.h"
#include <ESceneAIMapTools.h>




bool UIObjectList::LoadAiMAP()
{
	if (last_fileaimap.size() == 0)
		return false;

	string128 temp = { 0 };
	xr_strcat(temp, last_fileaimap.c_str());

	if (!FS.exist(temp))
	{
		Msg("File to Open: %s, not exist", temp);
		return false;
	}

	if (Scene->GetTool(OBJCLASS_AIMAP) )
	{
 		IReader* read = FS.r_open(temp);
		Msg("Open AI File: %s", temp);
		Scene->GetTool(OBJCLASS_AIMAP)->LoadStreamOFFSET(*read, vec_offset, ai_ignore_stractures);
		FS.r_close(read);
	}

	last_fileaimap.clear();

	return false;
}


void UIObjectList::SelectAIMAPFile()
{
	xr_string file;
	if (EFS.GetOpenName(EDevice.m_hWnd, _import_, file))
	{
		last_fileaimap = file;
	}
}


void UIObjectList::ExportAIMap(Fbox* box, LPCSTR name)
{
 	if (box)
	{
		ESceneAIMapTool* ai_tool = (ESceneAIMapTool*)Scene->GetTool(OBJCLASS_AIMAP);

		if (ai_tool)
		{
			string_path p;
			sprintf(p, "%s_ai", name);
			IWriter* ai_map = FS.w_open_ex(p);
			ai_map->open_chunk(2);
			int cnt = 0;
			for (auto node : ai_tool->Nodes())
			{
				if (box->contains(node->Pos))
					cnt++;
			}

			ai_map->w_u32(cnt);
			for (auto node : ai_tool->Nodes())
			{
				if (!use_outside_box && box->contains(node->Pos) || use_outside_box && !box->contains(node->Pos))
					ai_map->w_fvector3(node->Pos);
			}

			ai_map->close_chunk();
			FS.w_close(ai_map);
		}


	}
	else
	{
		if (Scene->GetTool(OBJCLASS_AIMAP) && Scene->GetTool(OBJCLASS_AIMAP)->Valid())
		{
			Msg("AI MAP");
			string_path path;
			string128 name_str = { 0 };
			xr_strcpy(name_str, "\\export_all_objects\\");
			xr_strcat(name_str, name);
			FS.update_path(path, _import_, name_str);
			xr_strcat(path, ".ai");

			IWriter* writer = FS.w_open_ex(path);
			ESceneAIMapTool* ai_tool = (ESceneAIMapTool*)Scene->GetTool(OBJCLASS_AIMAP);
			ai_tool->SaveStreamPOS(*writer);
			FS.w_close(writer);

		}
	}



	/*
	ESceneAIMapTool* ai_tool = (ESceneAIMapTool*)Scene->GetTool(OBJCLASS_AIMAP);

	if (ai_tool)
	{
		string_path p;
		sprintf(p, "%s_ai", temp_fn.c_str());
		IWriter* ai_map = FS.w_open_ex(p);
		ai_map->open_chunk(2);
		int cnt = 0;
		for (auto node : ai_tool->Nodes())
		{
			if (box.contains(node->Pos))
				cnt++;
		}

		ai_map->w_u32(cnt);
		for (auto node : ai_tool->Nodes())
		{
			if (!use_outside_box && box.contains(node->Pos) || use_outside_box && !box.contains(node->Pos))
				ai_map->w_fvector3(node->Pos);
		}

		ai_map->close_chunk();
		FS.w_close(ai_map);
	}
	*/
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


void UIObjectList::MergeAI_FromINI(CInifile* file)
{
	int id = 1;
	string16 section;
	sprintf(section, "aimap_%d", id);

	xr_vector <Fvector3>	result;

	while (file->section_exist(section))
	{
		Fvector offset = file->r_fvector3(section, "position");
		LPCSTR name_file = file->r_string(section, "name");

		string_path path;
		FS.update_path(path, _import_, name_file);

		xr_vector <Fvector3>	aimap = getAIPOS(path);
		for (auto pos : aimap)
		{
			pos.add(offset);
			result.push_back(pos);
		}

		Msg("LoadAIMap: %s, sizeof: %d, results: %d", path, aimap.size(), result.size());

		id++;
		sprintf(section, "aimap_%d", id);
	}

	if (file->section_exist("save_directory"))
	{
		string_path path;
		FS.update_path(path, _import_, file->r_string("save_directory", "path"));

		IWriter* write = FS.w_open(path);
		write->open_chunk(2);
		write->w_u32(result.size());
		for (auto pos : result)
			write->w_fvector3(pos);
		write->close_chunk();
		FS.w_close(write);
	}
}


xr_vector<Fvector3> UIObjectList::getAIPOS(LPCSTR file)
{
	IReader* read = FS.r_open(file);

	xr_vector<Fvector3> positions;

	if (read)
	{

		auto count = read->open_chunk(2);


		u32 size = read->r_u32();

		for (int i = 0; i < size; i++)
		{
			Fvector3 pos;
			read->r_fvector3(pos);
			positions.push_back(pos);
		}



		auto count_v3 = read->open_chunk(3);
		Msg("Load: %s, chunks: %d, v3: %d", file, count, count_v3);

	}

	FS.r_close(read);

	return positions;
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

		string128 file_end = { 0 };
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

 