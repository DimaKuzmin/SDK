#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "CustomObject.h"
#include <ESceneAIMapTools.h>
 
xr_vector<Fvector3> UIObjectList::getAIPOS(LPCSTR file)
{
	xr_vector<Fvector3> return_data;

	IReader* read = FS.r_open(file);
 	if (read)
	{
		auto count = read->open_chunk(2);
		u32 size = read->r_u32();
 		for (int i = 0; i < size; i++)
		{
			Fvector3 pos;
			read->r_fvector3(pos);
			return_data.push_back(pos);
		}
	}
 	FS.r_close(read);
 	return return_data;
}

bool UIObjectList::LoadAIMap()
{
	xr_string last_fileaimap;
	if (EFS.GetOpenName(EDevice.m_hWnd, _import_, last_fileaimap))
	{
 		string128 temp = { 0 };
		xr_strcat(temp, last_fileaimap.c_str());

		if (!FS.exist(temp))
		{
			Msg("File to Open: %s, not exist", temp);
			return false;
		}

		if (Scene->GetTool(OBJCLASS_AIMAP))
		{
			IReader* read = FS.r_open(temp);
			Msg("Open AI File: %s", temp);
			Scene->GetTool(OBJCLASS_AIMAP)->LoadStreamOFFSET(*read, vec_offset, ai_ignore_stractures);
			FS.r_close(read);
		}

		last_fileaimap.clear();
		return true;
	}
	else
		return false;
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