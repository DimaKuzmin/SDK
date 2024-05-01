#include "stdafx.h"
#include "UIWayTool.h"
#include "WayPoint.h"
#include "ESceneWayTools.h"

UIWayTool::UIWayTool()
{
	m_WayMode = true;
	m_AutoLink = true;
}

UIWayTool::~UIWayTool()
{
}

void UIWayTool::Draw()
{
    ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
    if (ImGui::TreeNode("Commands"))
    {
        ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
        {
            if (ImGui::RadioButton("Way Mode", m_WayMode))
            {
                LTools->SetTarget(OBJCLASS_WAY, 0);
                m_WayMode = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Way Point", m_WayMode == false))
            {
                LTools->SetTarget(OBJCLASS_WAY, 1);
                m_WayMode = false;
            }
        }
        ImGui::Separator();
        ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
        ImGui::TreePop();
    }
    ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
  
    if (ImGui::TreeNode("Link Command"))
    {
        ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
        {
            ImGui::Checkbox("Auto Link", &m_AutoLink);
           
            ImGui::PushItemWidth(-1);
            float size = float(ImGui::CalcItemWidth());
    
            if (ImGui::Button("Create 1-Link", ImVec2(size / 2, 0))) 
            {
                if (m_WayMode) {
                    ELog.DlgMsg(mtInformation, "Before editing enter Point Mode.");
                    return;
                }
                bool bRes = false;
                ObjectList lst;
                Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                // remove links
                for (ObjectIt it = lst.begin(); it != lst.end(); it++) {
                    ((CWayObject*)(*it))->RemoveLink();
                    bRes |= ((CWayObject*)(*it))->Add1Link();
                }
                if (bRes) Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }
            ImGui::SameLine(0, 2);
            if (ImGui::Button("Convert to 1-Link", ImVec2(size / 2, 0))) 
            {
                ObjectList lst;
                int cnt = Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                for (ObjectIt it = lst.begin(); it != lst.end(); it++)
                    ((CWayObject*)(*it))->Convert1Link();
                if (cnt)
                    Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }

            if (ImGui::Button("Create 2-Link", ImVec2(size / 2, 0))) 
            {
                if (m_WayMode)
                {
                    ELog.DlgMsg(mtInformation, "Before editing enter Point Mode.");
                    return;
                }
                bool bRes = false;
                ObjectList lst;
                Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                for (ObjectIt it = lst.begin(); it != lst.end(); it++)
                    bRes |= ((CWayObject*)(*it))->Add2Link();
                if (bRes) Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }
            ImGui::SameLine(0, 2);
            if (ImGui::Button("Convert to 2-Link", ImVec2(size / 2, 0))) 
            {
                ObjectList lst;
                int cnt = Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                for (ObjectIt it = lst.begin(); it != lst.end(); it++)
                    ((CWayObject*)(*it))->Convert2Link();
                if (cnt) Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }

            if (ImGui::Button("Invert Link", ImVec2(size / 2, 0))) 
            {
                if (m_WayMode) {
                    ELog.DlgMsg(mtInformation, "Before editing enter Point Mode.");
                    return;
                }
                ObjectList lst;
                int cnt = Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                for (ObjectIt it = lst.begin(); it != lst.end(); it++)
                    ((CWayObject*)(*it))->InvertLink();
                if (cnt) Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }
            ImGui::SameLine(0, 2);
            if (ImGui::Button("Remove Link", ImVec2(size / 2, 0)))
            {
                if (m_WayMode) {
                    ELog.DlgMsg(mtInformation, "Before editing enter Point Mode.");
                    return;
                }
                ObjectList lst;
                int cnt = Scene->GetQueryObjects(lst, OBJCLASS_WAY, 1, 1, 0);
                for (ObjectIt it = lst.begin(); it != lst.end(); it++)
                    ((CWayObject*)(*it))->RemoveLink();
                if (cnt) Scene->UndoSave();
                ExecCommand(COMMAND_UPDATE_PROPERTIES);
            }
            

        }
        ImGui::Separator();
        ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("SmartWorks"))
    {
        ImGui::InputText("#work_smart_name", object_prefix, 520);
        ImGui::InputInt("#work_smart_id", &work_id);
 
        auto prev_work = work_selected;
        if (ImGui::RadioButton("None", work_selected == eWorkNone))
            work_selected = eWorkNone;

        if (ImGui::RadioButton("Walker",    work_selected == eWorkWalker))
            work_selected = eWorkWalker;

        if (ImGui::RadioButton("Patrol",    work_selected == eWorkPatrol))
            work_selected = eWorkPatrol;

        if (ImGui::RadioButton("Guard",     work_selected == eWorkGuard))
            work_selected = eWorkGuard;

        if (ImGui::RadioButton("Sleeper",   work_selected == eWorkSleeper))
            work_selected = eWorkSleeper;

        if (ImGui::RadioButton("Animpoint", work_selected == eWorkAnimpoint))
            work_selected = eWorkAnimpoint;

        if (ImGui::RadioButton("Sniper",    work_selected == eWorkSniper))
            work_selected = eWorkSniper;

        if (ImGui::RadioButton("Surge",     work_selected == eWorkSurge))
            work_selected = eWorkSurge;

        ImGui::Text("WorkPoint Mode:");

        if (ImGui::RadioButton("Walk", work_selected_type == eWorkWalk))
            work_selected_type = eWorkWalk;
        if (ImGui::RadioButton("Look", work_selected_type == eWorkLook))
            work_selected_type = eWorkLook;


        //bool Modifyed = prev_work != work_selected;

        //if (Modifyed && work_selected != eWorkNone)
        if (ImGui::Button("Apply") && work_selected != eWorkNone)
        {
            ESceneWayTool* tool = (ESceneWayTool*) Scene->GetOTool(OBJCLASS_WAY);
            if (tool != nullptr)
            {
 
                CWayObject* objectsel = (CWayObject*)tool->LastSelected();
                if (objectsel != nullptr)
                {
                    shared_str name_prefix_work;
                    shared_str name_prefix_wtype;
                    switch (work_selected)
                    {
                        case UIWayTool::eWorkWalker:
                            name_prefix_work = "walker";
                            break;
                        case UIWayTool::eWorkPatrol:
                            name_prefix_work = "patrol";
                            break;
                        case UIWayTool::eWorkGuard:
                            name_prefix_work = "guard";
                            break;
                        case UIWayTool::eWorkSleeper:
                            name_prefix_work = "sleep";
                            break;
                        case UIWayTool::eWorkAnimpoint:
                            name_prefix_work = "animpoint";
                            break;
                        case UIWayTool::eWorkSniper:
                            name_prefix_work = "sniper";
                            break;
                        case UIWayTool::eWorkSurge:
                            name_prefix_work = "surge";
                            break;
                        default:
                            name_prefix_work = "not_implemented";
                            break;
                    }

                    switch (work_selected_type)
                    {
                        case UIWayTool::eWorkWalk:
                            name_prefix_wtype = "walk";
                            break;
                        case UIWayTool::eWorkLook:
                            name_prefix_wtype = "look";
                            break;
                        default:
                            name_prefix_wtype = "not_implemented";
                            break;
                    }

                    string256 worker_name;
                    sprintf(worker_name, "%s_%s_%d_%s", object_prefix, name_prefix_work.c_str(), work_id, name_prefix_wtype.c_str());
                    objectsel->SetName(worker_name);
                }
            }
             
        }
          
        ImGui::TreePop();
    }
}
