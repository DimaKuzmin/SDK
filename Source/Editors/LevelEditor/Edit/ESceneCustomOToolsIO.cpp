#include "stdafx.h"
#pragma hdrstop

#include "ESceneCustomOTools.h"
#include "scene.h"
#include "../XrECore/Editor/ui_main.h"
#include "builder.h"
#include "CustomObject.h"

// chunks
//----------------------------------------------------
static const u32 CHUNK_VERSION			= 0x0001;
static const u32 CHUNK_OBJECT_COUNT		= 0x0002;
static const u32 CHUNK_OBJECTS			= 0x0003;
static const u32 CHUNK_FLAGS			= 0x0004;
//----------------------------------------------------

bool ESceneCustomOTool::OnLoadSelectionAppendObject(CCustomObject* obj)
{
    string256 				buf;
    Scene->GenObjectName	(obj->FClassID,buf,obj->GetName());
    obj->SetName(buf);
    Scene->AppendObject		(obj, false);
    return					true;
}
//----------------------------------------------------

bool ESceneCustomOTool::OnLoadAppendObject(CCustomObject* O)
{
	Scene->AppendObject	(O,false);
    return true;
}
//----------------------------------------------------

bool ESceneCustomOTool::LoadSelection(IReader& F)
{
    int count					= 0;
	F.r_chunk					(CHUNK_OBJECT_COUNT,&count);

    SPBItem* pb 				= UI->ProgressStart(count,xr_string().sprintf("Loading %s(stream)...",ClassDesc()).c_str());
    Scene->ReadObjectsStream	(F,CHUNK_OBJECTS, EScene::TAppendObject(this, &ESceneCustomOTool::OnLoadSelectionAppendObject),pb);
    UI->ProgressEnd				(pb);

    return true;
}
//----------------------------------------------------

void ESceneCustomOTool::SaveSelection(IWriter& F)
{
	F.open_chunk	(CHUNK_OBJECTS);
    int count		= 0;
    for(ObjectIt it = m_Objects.begin();it!=m_Objects.end();++it)
    {
    	if ((*it)->Selected() && !(*it)->IsDeleted())
        {
	        F.open_chunk(count++);
    	    Scene->SaveObjectStream(*it,F);
        	F.close_chunk();
        }
    }
	F.close_chunk	();

	F.w_chunk		(CHUNK_OBJECT_COUNT,&count,sizeof(count));
}
//----------------------------------------------------

/*
xr_vector<shared_str> thread_work;
xrCriticalSection csLoad;
#include <thread>

void LoadThread(EScene* scene, CInifile* ini, SPBItem* pb)
{
    for (;;)
    {
        csLoad.Enter();

        if (thread_work.empty())
        {
            csLoad.Leave();
            break;
        }

        shared_str section = thread_work.back();
        thread_work.pop_back();

        csLoad.Leave();
        
        CCustomObject* obj = NULL;
        if (ini->section_exist(section))
        if (Scene->ReadObjectLTX(*ini, section.c_str(), obj))
        {
            if (!Scene->OnLoadAppendObject(obj))
                xr_delete(obj);
        }

        pb->Inc();
    }
} */
 
bool ESceneCustomOTool::LoadLTX(CInifile& ini)
{
	inherited::LoadLTX	(ini);

    u32 count			= ini.r_u32("main", "objects_count");

	SPBItem* pb 		= UI->ProgressStart(count,xr_string().sprintf("Loading %s(ltx)...",ClassDesc()).c_str());

    u32 i				= 0;
    string128			buff;
    CTimer t;
    t.Start();

    for(i=0; i<count; ++i)
    { 
        sprintf				(buff, "object_%d", i);
    
        CCustomObject* obj	= NULL;
        if (ini.section_exist(buff))
        if( Scene->ReadObjectLTX(ini, buff, obj) )
        {
            if (!OnLoadAppendObject(obj))
                xr_delete(obj);
        }

        pb->Inc();
       
        /* thread_work.push_back(buff);  */
    }
     /*
    std::thread* th[8];
    for (int i = 0; i < 8; i++)
        th[i] = new std::thread(LoadThread, Scene, &ini, pb);

    for (int i = 0; i < 8; i++)
        th[i]->join();

    Msg("Tool: %d, time: %u", FClassID, t.GetElapsed_ticks());


	
    */

    UI->ProgressEnd(pb);
    return true;
}

bool ESceneCustomOTool::LoadStream(IReader& F)
{
	inherited::LoadStream		(F);

    int count					= 0;
	F.r_chunk					(CHUNK_OBJECT_COUNT,&count);

    SPBItem* pb 				= UI->ProgressStart(count,xr_string().sprintf("Loading %s...",ClassDesc()).c_str());
    Scene->ReadObjectsStream	(F,CHUNK_OBJECTS, EScene::TAppendObject(this, &ESceneCustomOTool::OnLoadAppendObject),pb);
    UI->ProgressEnd				(pb);

    return true;
}
//----------------------------------------------------

void ESceneCustomOTool::SaveLTX(CInifile& ini, int id)
{
	inherited::SaveLTX	(ini, id);

	u32 count			= 0;
    for(ObjectIt it=m_Objects.begin(); it!=m_Objects.end(); ++it)
	{
    	CCustomObject* O = (*it);
        if(O->save_id!=id)
        	continue;
            
    	if (O->IsDeleted() || O->m_CO_Flags.test(CCustomObject::flObjectInGroup) )
        	continue;
            
        string128				buff;
        sprintf					(buff,"object_%d",count);
        Scene->SaveObjectLTX	(*it,  buff, ini);
        count++;
	}

	ini.w_u32			("main", "objects_count", count);
}

void ESceneCustomOTool::SaveStream(IWriter& F)
{
	inherited::SaveStream	(F);

    int Objcount		= 0;

	F.open_chunk		(CHUNK_OBJECTS);
    int count			= 0;
    for(ObjectIt it = m_Objects.begin();it!=m_Objects.end();++it)
	{
    	if ( (*it)->IsDeleted() || (*it)->m_CO_Flags.test(CCustomObject::flObjectInGroup) )
        continue;

        F.open_chunk			(count++);
        Scene->SaveObjectStream	(*it,F);
        F.close_chunk			();
    }
	F.close_chunk	();

	F.w_chunk		(CHUNK_OBJECT_COUNT,&Objcount,sizeof(Objcount));
}
//----------------------------------------------------

bool ESceneCustomOTool::Export(LPCSTR path)
{
	return true;
}
//----------------------------------------------------
 
bool ESceneCustomOTool::ExportGame(SExportStreams* F)
{
	bool bres=true;
    for(ObjectIt it = m_Objects.begin();it!=m_Objects.end();it++)
        if (!(*it)->ExportGame(F)) bres=false;
	return bres;
}
//----------------------------------------------------

bool ESceneCustomOTool::ExportStatic(SceneBuilder* B, bool b_selected_only)
{
	return B->ParseStaticObjects(m_Objects, NULL, b_selected_only);
}
 BOOL GetStaticCformData   ( ObjectList& lst, mesh_build_data &data, bool b_selected_only );
bool ESceneCustomOTool::GetStaticCformData( mesh_build_data &data, bool b_selected_only ) //b_vertex* verts, int& vert_cnt, int& vert_it,b_face* faces, int& face_cnt, int& face_it,
{
      return    ::GetStaticCformData(  m_Objects, data, b_selected_only );
}
//----------------------------------------------------
 
