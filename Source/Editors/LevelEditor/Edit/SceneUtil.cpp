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
	if(!name)
    return NULL;
    
	CCustomObject* object = 0;
    if (classfilter==OBJCLASS_DUMMY)
    {
        SceneToolsMapPairIt _I = m_SceneTools.begin();
        SceneToolsMapPairIt _E = m_SceneTools.end();
        for (; _I!=_E; ++_I)
        {
            ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(_I->second);

            if (mt&&(0!=(object=mt->FindObjectByName(name))))
            	return object;
        }
    }else{
        ESceneCustomOTool* mt = GetOTool(classfilter); VERIFY(mt);
        if (mt&&(0!=(object=mt->FindObjectByName(name)))) return object;
    }
    return object;
}

CCustomObject* EScene::FindObjectByName( LPCSTR name, CCustomObject* pass_object )
{
	CCustomObject* object = 0;
    SceneToolsMapPairIt _I = m_SceneTools.begin();
    SceneToolsMapPairIt _E = m_SceneTools.end();
    for (; _I!=_E; _I++){
        ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(_I->second);
        if (mt&&(0!=(object=mt->FindObjectByName(name,pass_object)))) return object;
    }
    return 0;
}

bool EScene::FindDuplicateName()
{
// find duplicate name
    SceneToolsMapPairIt _I = m_SceneTools.begin();
    SceneToolsMapPairIt _E = m_SceneTools.end();
    for (; _I!=_E; _I++){
        ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(_I->second);
        if (mt){
        	ObjectList& lst = mt->GetObjects(); 
            for(ObjectIt _F = lst.begin();_F!=lst.end();_F++)
                if (FindObjectByName((*_F)->GetName(), *_F)){
                    ELog.DlgMsg(mtError,"Duplicate object name already exists: '%s', Class: %d, Ref: %s, POS[%f][%f][%f]",(*_F)->GetName(), (mt->FClassID), (*_F)->RefName(), VPUSH((*_F)->GetPosition()) );
                    return true;
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


