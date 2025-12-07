//----------------------------------------------------
// file: SceneUtil.cpp
//----------------------------------------------------

#include "stdafx.h"
#pragma hdrstop

#include "Scene.h"
#include "ELight.h"
#include "SceneObject.h"
#include "ui_leveltools.h"

//----------------------------------------------------
CCustomObject* EScene::FindObjectByName( LPCSTR name, ObjClassID classfilter )
{
    if (!name)
        return NULL;

	CCustomObject* object = 0;

    if (classfilter==OBJCLASS_DUMMY)
    {
        for (auto& tool : m_SceneTools)
        {
            ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(tool.second);
           
            if (mt)
            {
               object = mt->FindObjectByName(name);
            }
           
        }
    }
    else
    {
        ESceneCustomOTool* mt = GetOTool(classfilter); VERIFY(mt);
      
        if (mt)
        {
             object = mt->FindObjectByName(name);
        }
    }

    return object;
}

#include <execution>

CCustomObject* EScene::FindObjectByName( LPCSTR name, CCustomObject* pass_object )
{
    for (auto& tool : m_SceneTools)
    {   
        ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(tool.second);
        if (mt != nullptr)
        {
            auto O = mt->FindObjectByName(name, pass_object);
            if (O)
            {
                return O;
            }
        }
    } 
    return nullptr;
}

bool EScene::FindDuplicateName()
{    
    for (auto& tool : m_SceneTools)
    {
        ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(tool.second);
     
        if (mt)
        {
            xr_vector<CCustomObject*> objects;
            xr_unordered_map <size_t, CCustomObject*> names;
            for (auto F : mt->GetObjects())
            {
                objects.push_back(F);
                F->GenHash();
              
                if (names[F->GetHash()] != nullptr)
                {
                    Msg("Finded Dublicate Hash: %llu, NameObject: %s", F->GetHash(), F->GetName());
                }
                else 
                    names[F->GetHash()] = F;

            }

            for (auto F : mt->GetObjects())
            {
                if (names[F->GetHash()] && F != names[F->GetHash()])
                {
                    ELog.DlgMsg(mtError, "Duplicate object name already exists: '%s', Class: %d, Ref: %s, POS[%f][%f][%f]", F->GetName(), (mt->FClassID), F->RefName(), VPUSH(F->GetPosition()));
                    return true;
                }
                else
                {
                   // Msg("Check Object No finded Dublicate: %s", names[F->GetHash()]->FName.c_str());
                }
            }
        }
    }
    return false;
}

void EScene::GenObjectName(ObjClassID cls_id, char* buffer, const char* pref)
{
    for (int i = 0; true; i++)
    {
        bool result;
        xr_string temp;
        if (pref && pref[0])
        {
            if (i == 0)
            {
                temp = pref;
            }
            else
            {
                temp.sprintf("%s_%02d", pref, i - 1);
            }
        }
        else
        {
            temp.sprintf("%02d", i );
        }
       
        FindObjectByNameCB(temp.c_str(), result);
        if (!result)
        {
            xr_strcpy(buffer, 256, temp.c_str());
            return;
        }
    }
    /*ESceneCustomOTool* ot = GetOTool(cls_id); VERIFY(ot);
    xr_string result	= FHelper.GenerateName(pref&&pref[0]?pref:ot->ClassName(),4,fastdelegate::bind<TFindObjectByName>(this,&EScene::FindObjectByNameCB),true,true);
    strcpy				(buffer,result.c_str());*/
}
//------------------------------------------------------------------------------


