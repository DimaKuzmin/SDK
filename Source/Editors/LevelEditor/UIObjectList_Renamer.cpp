#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "SceneObject.h"

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

}

bool sort_list(CCustomObject* obj1, CCustomObject* obj2)
{
	if (obj1->RefName() && obj2->RefName())
		if (xr_strcmp(obj1->RefName(), obj2->RefName()) < 0)
			return true;

	return false;
};

void UIObjectList::RenameALLObjectsToObject()
{
	//for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
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


		if (ot->FClassID == OBJCLASS_SCENEOBJECT)  // ot->FClassID == OBJCLASS_SPAWNPOINT 
		{
			int id = 1;

			xr_map<LPCSTR, u16> map_names_ref;

			ObjectList list = ot->GetObjects();
			for (auto item : list)
			{
				//	Msg("Name %s", item->RefName());

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

			if (use_prefix_refname)
			{
				CSceneObject* scene = smart_cast<CSceneObject*>(item);
				if (scene)
					sprintf(name, "%s_%s_%d", scene->RefName(), &rename_prefix_name, i);
			}
			else
				sprintf(name, "%s_%d", &rename_prefix_name, i);

			item->SetName(name);

			i++;
		}
	}
}