#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "CustomObject.h"
#include <ESceneAIMapTools.h>
#include "SceneObject.h"


bool UIObjectList::ExportDir(xr_string& dir)
{
	if (EFS.GetSaveName(_import_, dir))
		return true;

	return false;
}

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

void UIObjectList::ExportInsideBox()
{
	Fbox box;
	box.min = vec_box_min;
	box.max = vec_box_max;

	xr_string temp_fn = "";

	if (EFS.GetSaveName(_import_, temp_fn))
	{
		CInifile file(temp_fn.c_str(), false, false, false);

		int i = 0;

		for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
		{
			ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
			if (!ot)
				continue;

			if (ot->FClassID == OBJCLASS_DUMMY)
				continue;

			if (!ot->can_use_inifile())
				continue;

			ObjectList& lst = ot->GetObjects();
			for (auto obj : lst)
			{
				Fbox newB;
				obj->GetBox(newB);

				if (!use_outside_box && box.contains(newB) || use_outside_box && !box.contains(newB))
				{
					string32 buffer = { 0 };
					sprintf(buffer, "object_%d", i);
					Scene->SaveObjectLTX(obj, buffer, file);
					++i;
				}
			}
		}
		file.save_as(temp_fn.c_str());

		string_path p;
		sprintf(p, "%s_ai", temp_fn.c_str());
		ExportAIMap(&box, p);
	}
}

void UIObjectList::SetListToMove()
{
	Fbox box;
	box.min = vec_box_min;
	box.max = vec_box_max;

	objects_to_move.clear();

	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		if (!ot)
			continue;
		 
		ObjectList& lst = ot->GetObjects();

		for (auto obj : lst)
		{
			if (obj && box.contains(obj->GetPosition()))
			{
				objects_to_move[obj] = obj->GetPosition();
			}
		}
	}
}

void UIObjectList::AddSelectedToMove()
{
	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		if (!ot)
			continue;

		ObjectList& lst = ot->GetObjects();

		for (auto obj : lst)
		{
			if (obj->Selected())
			{
				objects_to_move[obj] = obj->GetPosition();
			}
		}
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
			sprintf(name, "%s_%s", Scene->m_LevelOp.m_FNLevelPath.c_str(), obj->GetName());
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

	ExportAIMap(0, Scene->m_LevelOp.m_FNLevelPath.c_str());
}



// CLEARING (MAYBY NEW FILE)
void UIObjectList::RemoveAllInsideBox()
{
	Fbox box;
	box.min = vec_box_min;
	box.max = vec_box_max;

	xr_vector<CCustomObject*> object_to_destroy;

 	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		if (!ot)
			continue;

		ObjectList& lst = ot->GetObjects();

		for (auto obj : lst)
		{
			Fbox newB;
			obj->GetBox(newB);

 			if ( obj && box.contains(newB) )
			{
				object_to_destroy.push_back(obj);
			}
		}
	}

	for (auto obj : object_to_destroy)
	{
		if (Scene != nullptr)
		{
			obj->DeleteThis();
			Scene->RemoveObject(obj, false, true);
			
			LPCSTR name = obj->FName.c_str();
			Msg("remove : %s", name);
		}
	}

}

 
void UIObjectList::BboxSelectedObject()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

	//Fbox box_all;
	//Scene->GetBox(box_all, LTools->CurrentClassID());
	
	Fbox box_all;
	bool boxIsset = false;
	ObjectList& list = ot->GetObjects();
	for (auto item : list)
	{
 		if (!item->Selected())
			continue;

		Fbox box;
		bool isRead = item->GetBox(box);
		if (isRead)
		{
			if (!boxIsset)
			{
				box_all.set(box.min, box.max);
				boxIsset = true;
			}
			else
			{
				box_all.modify(box.min);
				box_all.modify(box.max);
			}
		}
		
		Msg("OBJECT BBOX x[%f][%f]", box.x1, box.x2);
		Msg("OBJECT BBOX z[%f][%f]", box.z1, box.z2);
		Msg("OBJECT BBOX y[%f][%f]", box.y1, box.y2);
	}

	vec_box_min = box_all.min;
	vec_box_max = box_all.max;

	

	Msg("Selected BBOX x[%f][%f]", box_all.x1, box_all.x2);
	Msg("Selected BBOX z[%f][%f]", box_all.z1, box_all.z2);
	Msg("Selected BBOX y[%f][%f]", box_all.y1, box_all.y2);


}

void UIObjectList::SelectObjectsInsideBox()
{
	Fbox box;
	box.min = vec_box_min;
	box.max = vec_box_max;

 
	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		if (!ot)
			continue;

		ObjectList& lst = ot->GetObjects();

		for (auto obj : lst)
		{
			Fbox newB;
			obj->GetBox(newB);

			if (obj && box.contains(newB))
			{
				obj->Select(true);
			}
		}
	}
	 


}


