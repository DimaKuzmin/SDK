#include "stdafx.h"
#include "UILods.h"
 #include "..\XrECore\Editor\Library.h"


UILods* UILods::Form_Tree = nullptr;
 

void UILods::Show()
{

	if (Form_Tree == nullptr)
	{	
 		Form_Tree = xr_new< UILods>();
	}
}

void UILods::Close()
{
 	xr_delete(Form_Tree);
}

void UILods::Update()
{ 
 	if (Form_Tree)
	{
		if (!Form_Tree->IsClosed())
		{
			Form_Tree->Draw();
		}
		else
		{
 			xr_delete(Form_Tree);
		}
	}
}


bool UILods::Refresh_ObjectsRef()
{
	refs_objects_vec.clear();

	ListItemsVec items;
    FS_FileSet lst;
    if (Lib.GetObjects(lst)) 
    {
        FS_FileSetIt	it = lst.begin();
        FS_FileSetIt	_E = lst.end();
        for (; it != _E; it++) 
        {
			//Msg("File: %s", it->name.c_str());
            
			//if (strstr(it->name.c_str(), "trees\\") || strstr(it->name.c_str(), "new_trees\\"))
			if (strstr(it->name.c_str(), search_prefix_o))
				refs_objects_vec.push_back(it->name.c_str());


			//xr_string fn;
            //LHelper().CreateItem(items, it->name.c_str(), 0, ListItem::flDrawThumbnail, 0);
        }
    }

	return true;
}

void ThreadWork(UILods * logs)
{
	for (auto ref : logs->refs_objects_vec)
	{
		string128 name;
		sprintf(name, "%s.object", ref.c_str());

		string_path ref_path;
		FS.update_path(ref_path, _objects_, name);

			

		if (ATools->Load(ref_path))
		{
			Msg("Lod: %s", ref_path);

			//EPrefs->AppendRecentFile(ref_path);
			//ExecCommand(COMMAND_UPDATE_CAPTION);
			//ExecCommand(COMMAND_UPDATE_PROPERTIES);


			ATools->GenerateLOD(1);

			while(ATools->m_Flags.is(CActorTools::flGenerateLODHQ))
			{
 				Sleep(100);
			}

			Sleep(1000);

			ATools->Clear();
		}
		else 
		{
			Msg("Cant Load: %s", ref_path);
		}

	}
}

#include <thread>


void UILods::Draw()
{
	if (!ImGui::Begin("UI LODS GENERATOR", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}

	if (ImGui::BeginChild("#ListObjects_REFS", ImVec2(500, 350) ) )
	{

		ImGui::ListBoxHeader("##list_objects", ImVec2(500, 200));
		{
			for (int i = 0; i < refs_objects_vec.size(); ++i) 
			{
				if (ImGui::Selectable(refs_objects_vec[i].c_str(), current_item_object == i, 0, ImVec2(350, 14))) 
				{
					current_item_object = i;
				}
			}
		}
		ImGui::ListBoxFooter();

		ImGui::InputText("#serch", search_prefix_o, 520);
 
		if (m_RealTexture_replace)
		{
			//ImGui::SetCursorPos(ImVec2(1100, PosY));
			ImGui::Image(m_RealTexture_replace, ImVec2(125, 125));
		}
		 
	}
	ImGui::EndChild();

	if (last_selected_object != current_item_object)
	{
		if (refs_objects_vec.size() > 0)
		{
			last_selected_object = current_item_object;
			
			auto * m_Thm = ImageLib.CreateThumbnail(refs_objects_vec[current_item_object].c_str(), EImageThumbnail::ETObject);
			m_RealTexture_replace = nullptr;
			if (m_Thm)
			{	
				m_Thm->Update(m_RealTexture_replace);	

				Msg("Update Tex:   m_RealTexture_replace : %d", m_RealTexture_replace);
			}
		}
	}

	if (ImGui::Button("Refresh"))
 		Refresh_ObjectsRef();

	if (ImGui::Button("GenLoads"))
	{
 		//std::thread* th = new std::thread(ThreadWork, this);
		//th->detach();
		gen_lods = true;
		last_gen_object = 0;
		PB = UI->ProgressStart(refs_objects_vec.size(), "Generate Lods" );
	}

	if (last_gen_object == refs_objects_vec.size())
	{
		gen_lods = false;
		last_gen_object = -1;
		UI->ProgressEnd(PB);
	}

	ImGui::Checkbox("HQ", &Quallyty);
	 

	if (gen_lods && ( !ATools->m_Flags.is(CActorTools::flGenerateLODHQ) && !ATools->m_Flags.is(CActorTools::flGenerateLODLQ) )  && last_gen_object < refs_objects_vec.size())
	{
		string128 name;
		sprintf(name, "%s.object", refs_objects_vec[last_gen_object].c_str());

		string_path ref_path;
		FS.update_path(ref_path, _objects_, name);
 
		if (ATools->Load(ref_path))
		{
			string256 tmp;
			sprintf(tmp, "Lod[%d]<[%d]: %s", last_gen_object, refs_objects_vec.size() , ref_path);
 			//PB->Inc();
			PB->Info(tmp);

			ATools->GenerateLOD(Quallyty);

			// while(ATools->m_Flags.is(CActorTools::flGenerateLODHQ))
  			//	Sleep(100);
 
			//Sleep(1000);

			//ATools->Clear();
		}
		else 
		{
			Msg("Cant Load: %s", ref_path);
		}

		last_gen_object++;
	}
	

 
	ImGui::End();
}