#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "SceneObject.h"
 
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

			Fbox box;
			item->GetBox(box);

			Fvector center;
			box.getcenter(center);

			ini_file->w_fvector3(item->GetName(), "position", item->GetPosition());
 			ini_file->w_fvector3(item->GetName(), "box_min", box.min);
			ini_file->w_fvector3(item->GetName(), "box_max", box.max);
 			ini_file->w_fvector3(item->GetName(), "box_center", center);

			CSceneObject* object = smart_cast<CSceneObject*>(item);
			if (object)
				ini_file->w_fvector3(item->GetName(), "ref_pos", object->m_pReference->ObjectXFORM().c);
		}

		ini_file->save_as(file.c_str());
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
				edit_obj->use_global_pos = use_global_position;
 
				edit_obj->a_vPosition = obj->GetPosition();
				edit_obj->a_vRotate = obj->GetRotation();

				edit_obj->Save(*write);
				write->close_chunk();
			}

		}


		FS.w_close(write);
	}
}