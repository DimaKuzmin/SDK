#include "stdafx.h"
#include "UIObjectList.h"
#include "Edit\scene.h"
#include "Edit\ESceneCustomOTools.h"
#include "Edit\CustomObject.h"
#include "Edit\GroupObject.h"

UIObjectList* UIObjectList::Form = nullptr;
UIObjectList::UIObjectList()
{
	m_Mode = M_Visible;
	m_Filter[0] = 0;
	m_SelectedObject = nullptr;
}

UIObjectList::~UIObjectList()
{
}

 
void UIObjectList::Draw()
{
	int offset_y = LTools->CurrentClassID() == OBJCLASS_SPAWNPOINT ? 180 : 0;

	int offset_x = 100;

 	ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(400 + offset_x, 400 + offset_y));
	 

	if (!ImGui::Begin("Object List", &bOpen))
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}

	{
			ImGui::BeginGroup();
		
			if (ImGui::BeginChild("Left", ImVec2(-200 - offset_x, -ImGui::GetFrameHeight()-8 - offset_y), true))
			{
				DrawObjects();
			}
			ImGui::EndChild();

			ImGui::SetNextItemWidth(-200 - offset_x);
			ImGui::InputText("##value", m_Filter, sizeof(m_Filter));

			if (offset_y > 0)
 			{ 
				ListBoxForTypes();
			}
		 
		
		
		ImGui::EndGroup();
	}
	
	ImGui::SameLine();

	if (ImGui::BeginChild("Right", ImVec2(200 + offset_x, 0)))
	{
		
		if (ImGui::RadioButton("All", m_Mode == M_All))
		{
			m_Mode = M_All;
		}

		if (ImGui::RadioButton("Visible Only", m_Mode == M_Visible))
		{
			m_Mode = M_Visible;

		}
		if (ImGui::RadioButton("Invisible Only", m_Mode == M_Inbvisible))
		{
			m_Mode = M_Inbvisible;
		}
		//ImGui::Separator();

		if (ImGui::Button("Show Selected", ImVec2(-1, 0)))
		{
			if (m_SelectedObject)
			{
				m_SelectedObject->Show(true);
				m_SelectedObject->Select(true);
			}
		}

		if (ImGui::Button("Hide Selected", ImVec2(-1, 0)))
		{
			if (m_SelectedObject)
			{
				m_SelectedObject->Show(false);
			}
		}
 
		UpdateUIObjectList();
	}
	ImGui::EndChild();

	ImGui::PopStyleVar(1);
	ImGui::End();
}

void UIObjectList::Update()
{
	if (Form)
	{
		if (!Form->IsClosed())
		{
			Form->Draw();
		}
		else
		{
			xr_delete(Form);
		}
	}

}

void UIObjectList::Show()
{
	if (Form == nullptr)
		Form = xr_new< UIObjectList>();
}

void UIObjectList::Close()
{
	xr_delete(Form);
}

void UIObjectList::DrawObjects()
{	
//	Msg("CurrentTOOLS %d", LTools->CurrentClassID() );
	m_cur_cls = LTools->CurrentClassID();
	string1024				str_name;

	for (SceneToolsMapPairIt it = Scene->FirstTool(); it != Scene->LastTool(); ++it)
	{
		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(it->second);
		if (ot && ((m_cur_cls == OBJCLASS_DUMMY) || (it->first == m_cur_cls)))
		{
			if (it->first == OBJCLASS_DUMMY)
				continue;
			ObjectList& lst = ot->GetObjects();
			ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
		
			if (ImGui::TreeNode("floder", ("%ss", it->second->ClassDesc())))
			{
				if (OBJCLASS_GROUP == it->first)
				{
					for (ObjectIt _F = lst.begin(); _F != lst.end(); ++_F)
					{
						switch (m_Mode)
						{
						case UIObjectList::M_All:
							break;
						case UIObjectList::M_Visible:
							if (!(*_F)->Visible())continue;
							break;
						case UIObjectList::M_Inbvisible:
							if ((*_F)->Visible())continue;
							break;
						default:
							break;
						}
						{
							strcpy(str_name, ((CGroupObject*)(*_F))->GetName());
						}
						DrawObject(*_F, str_name);
						ImGui::PushID(str_name);
						if (ImGui::TreeNode(str_name))
						{
							ObjectList 					grp_lst;

							((CGroupObject*)(*_F))->GetObjects(grp_lst);

							for (ObjectIt _G = grp_lst.begin(); _G != grp_lst.end(); _G++)
							{
								DrawObject(*_G, 0);
							}
							ImGui::TreePop();
						}
						ImGui::PopID();
					}
				}
				else
				{
					bool FindSelectedObj = false;
					for (ObjectIt _F = lst.begin(); _F != lst.end(); ++_F)
					{
						switch (m_Mode)
						{
						case UIObjectList::M_All:
							break;
						case UIObjectList::M_Visible:
							if (!(*_F)->Visible())continue;
							break;
						case UIObjectList::M_Inbvisible:
							if ((*_F)->Visible())continue;
							break;
						default:
							break;
						}
						DrawObject(*_F,0);
						FindSelectedObj = FindSelectedObj | (*_F) == m_SelectedObject;
					}
					if (!FindSelectedObj)
						m_SelectedObject = nullptr;

				}
				ImGui::TreePop();
			}
		
		}
	}
}

void UIObjectList::DrawObject(CCustomObject* obj, const char* name)
{ 
	if (m_Filter[0] && obj->GetName() != 0)
	if (strstr(obj->GetName(), m_Filter) == 0)
		return;

	if (!CheckNameForType(obj))
		return;

	ImGuiTreeNodeFlags Flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

	if (obj->Selected())
	{
		Flags |= ImGuiTreeNodeFlags_Bullet;
	}

	if (m_SelectedObject == obj)
	{
		Flags |= ImGuiTreeNodeFlags_Selected;
	}	 

	if (name)
		ImGui::TreeNodeEx(name, Flags);
	else
		ImGui::TreeNodeEx(obj->GetName(), Flags);

	if (ImGui::IsItemClicked())
	{
		if (m_SelectedObject != obj)
		{
			obj->Select(true);

			m_SelectedObject = obj;
		}
	}

	if (ImGui::GetIO().KeyAlt)
		obj->Select(false);

		
	if (ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift)
	{
		if (!obj->Selected())
		{
			obj->Select(true);
		}	 
 	}
 
}



