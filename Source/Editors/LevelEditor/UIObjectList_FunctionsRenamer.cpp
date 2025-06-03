#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "SceneObject.h"


void UIObjectList::FindALL_Duplicate()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

 	for (auto F : ot->GetObjects())
	{
		if (ot->FindObjectByName(F->GetName(), F) != 0)
		{
			Msg("Finded Dublicate NameObject: %s", F->GetName());
		}
	}	 
}


void UIObjectList::CheckDuplicateNames()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
 	for (auto item : ot->GetObjects())
	{
 		if (ot->FindObjectByName(item->GetName(), item) != 0)
		{
			u32 IDX = 0;
 			string64 test_name;
			sprintf(test_name, "%s_%d", item->GetName(), IDX);
 			while (ot->FindObjectByName(test_name, 0) != 0)
			{
				sprintf(test_name, "%s_%d", item->GetName(), IDX);
				IDX++;
 			}

			Msg("Rename Obj %s to %s", item->GetName(), test_name);

			item->SetName(test_name);
		}
	}
}

bool sort_list(CCustomObject* obj1, CCustomObject* obj2)
{
	if (obj1->RefName() && obj2->RefName())
		if (xr_strcmp(obj1->RefName(), obj2->RefName()) < 0)
			return true;

	return false;
};

void UIObjectList::RenameALLObjectsToSpawns()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID())); //it

	if (ot->FClassID == OBJCLASS_SPAWNPOINT)   
	{
		int id = 1;
		xr_map<LPCSTR, u16> map_names_ref;

		ObjectList list = ot->GetObjects();
		for (auto item : list)
		{
			if (item->Selected())
			{
				string256 name_new = { 0 };
				sprintf(name_new, "%s_%4d", item->GetName(), id);
 				item->SetName(name_new);
				id++;
			}			
		}

		list.sort(sort_list);
	}

}

void UIObjectList::RenameALLObjectsToObject()
{
 	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID())); //it

	if (ot->FClassID == OBJCLASS_LIGHT ||
		ot->FClassID == OBJCLASS_GLOW ||
		ot->FClassID == OBJCLASS_SECTOR ||
		ot->FClassID == OBJCLASS_PORTAL ||
		ot->FClassID == OBJCLASS_PS
		)
	{
		ObjectList list = ot->GetObjects();
		string256 name_prefix = { 0 }, tmp;
		xr_string tool_class;

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


	if (ot->FClassID == OBJCLASS_SCENEOBJECT)  
	{
		int id = 1;
 		xr_map<LPCSTR, u16> map_names_ref;

		ObjectList list = ot->GetObjects();
		for (auto item : list)
		{
			string256 prefix = { 0 };
			if (item->RefName())
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

			string256 name_new = { 0 }, tmp;
			xr_strcat(name_new, prefix);
			xr_strcat(name_new, "_");
			xr_strcat(name_new, itoa(id, tmp, 10));

			item->SetName(name_new);
		}

		list.sort(sort_list);
	}
}

void UIObjectList::RenameSelectedObjectsPrefix()
{
	ESceneCustomOTool* base = Scene->GetOTool(LTools->CurrentClassID());
	int i = 0;
	for (auto item : base->GetObjects())
	{
		if (item->Selected())
		{
			string256 name;
			sprintf(name, "%s_%4d", &rename_prefix_name, i);
 			item->SetName(name);

			i++;
		}
	}
}

void UIObjectList::RenameSelectedObjects()
{
	ESceneCustomOTool* base = Scene->GetOTool(LTools->CurrentClassID());
	
	int i = 0;
	for (auto item : base->GetObjects())
	{
		if (item->Selected())
		{
			string256 name;
			sprintf(name, "%s_%4d", item->GetName(), i);

			item->SetName(name);

			i++;
		}
	}
}