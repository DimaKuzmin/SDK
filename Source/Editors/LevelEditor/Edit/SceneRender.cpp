#include "stdafx.h"
#pragma hdrstop

#include "Scene.h"
#include "SceneObject.h"


#ifdef USE_ARENA_ALLOCATOR
static const u32	s_arena_size = 32 * 1024 * 1024;
char* s_fake_array = nullptr;
doug_lea_allocator	g_render_lua_allocator(s_fake_array, s_arena_size, "render:lua");
#else // #ifdef USE_ARENA_ALLOCATOR
doug_lea_allocator	g_render_lua_allocator(0, 0, "render:lua");
#endif // #ifdef USE_ARENA_ALLOCATOR



//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#define RENDER_OBJECT(P,B)\
{\
    try{\
        (N->val)->RenderRoot(P,B);\
    }catch(...){\
        ELog.DlgMsg(mtError, "Please notify AlexMX!!! Critical error has occured in render routine!!! [Type B] - Tools: '%s' Object: '%s'",(N->val)->FParentTools->ClassName(),(N->val)->GetName());\
    }\
}
    
void  object_Normal_0(EScene::mapObject_Node *N)	 
{
    (N->val)->RenderRoot(0, false);
    //RENDER_OBJECT(0,false);
}
void  object_Normal_1(EScene::mapObject_Node *N)	
{
    (N->val)->RenderRoot(1, false);
    //RENDER_OBJECT(1,false);
}
void  object_Normal_2(EScene::mapObject_Node *N)	 
{
    (N->val)->RenderRoot(2, false);
    //RENDER_OBJECT(2,false); 
}
void  object_Normal_3(EScene::mapObject_Node *N)	 
{
    (N->val)->RenderRoot(3, false);
    //RENDER_OBJECT(3,false); 
}

//------------------------------------------------------------------------------
void  object_StrictB2F_0(EScene::mapObject_Node *N)
{
    //RENDER_OBJECT(0,true);
    (N->val)->RenderRoot(0, true);
}
void  object_StrictB2F_1(EScene::mapObject_Node *N)
{
    //RENDER_OBJECT(1,true);
    (N->val)->RenderRoot(1, true);
}

void  object_StrictB2F_2(EScene::mapObject_Node *N)
{
    //RENDER_OBJECT(2,true);
    (N->val)->RenderRoot(2, true);
}

void  object_StrictB2F_3(EScene::mapObject_Node *N)
{
    (N->val)->RenderRoot(3, true);
    //RENDER_OBJECT(3,true);
}
//------------------------------------------------------------------------------

 

void EScene::RenderSky(const Fmatrix& camera)
{
	if( !valid() )	return;

//	draw sky
/*
//.
	if (m_SkyDome&&fraBottomBar->miDrawSky->Checked){
        st_Environment& E = m_LevelOp.m_Envs[m_LevelOp.m_CurEnv];
        m_SkyDome->GetPosition() = camera.c;
        m_SkyDome->UpdateTransform(true);
		EDevice.SetRS(D3DRS_TEXTUREFACTOR, E.m_SkyColor.get());
    	m_SkyDome->RenderSingle();
	    EDevice.SetRS(D3DRS_TEXTUREFACTOR,	0xffffffff);
    }
*/
}
//------------------------------------------------------------------------------

struct tools_rp_pred 
{
    IC bool operator()(ESceneToolBase* x, ESceneToolBase* y) const
    {	return x->RenderPriority()<y->RenderPriority();	}
};

#define DEFINE_MSET_PRED(T,N,I,P)	typedef xr_multiset< T, P > N;		typedef N::iterator I;

DEFINE_MSET_PRED(ESceneToolBase*,SceneMToolsSet,SceneMToolsIt,tools_rp_pred);
DEFINE_MSET_PRED(ESceneCustomOTool*,SceneOToolsSet,SceneOToolsIt,tools_rp_pred);

#include "ppl.h"

void RenderScene(SceneMToolsSet& scene, int P, bool B)
{
    SceneMToolsIt s_it = scene.begin();
    SceneMToolsIt s_end = scene.end();

    //for (; s_it != s_end; s_it++)

    // concurrency::parallel_for_each(s_it, s_end, [&](ESceneToolBase* tool)

    for (; s_it != s_end; s_it++)
    {
        EDevice.SetShader(B ? EDevice.m_SelectionShader : EDevice.m_WireShader);
        RCache.set_xform_world(Fidentity);
        (*s_it)->OnRenderRoot(P, B);

    };

    // });

     
}

 
void EScene::UpdateRenderList(void* tools)
{
    // extract and sort object tools

    SceneOToolsSet* object_tools = (SceneOToolsSet*)tools;
    mapRenderObjects.clear();
  
    SceneOToolsIt t_it = object_tools->begin();
    SceneOToolsIt t_end = object_tools->end();
    for (; t_it != t_end; t_it++)
    {
        ObjectList& lst = (*t_it)->GetObjects();
        ObjectIt o_it = lst.begin();
        ObjectIt o_end = lst.end();
        for (; o_it != o_end; o_it++)
        {
            if ((*o_it)->Visible() && (*o_it)->IsRender())
            {
                float distSQ = EDevice.vCameraPosition.distance_to_sqr((*o_it)->FPosition);
                mapRenderObjects.insertInAnyWay(distSQ, *o_it);
            }
        }
    }
   
};



ObjectList list_mt_work[4];
 
void MT_Render(int TH_ID)
{
    
}
/*
#define PROFILE_ECAPTURE_START OPTICK_START_CAPTURE
#define PROFILE_ECAPTURE_STOP  OPTICK_STOP_CAPTURE
#define PROFILE_ECAPTURE_SAVE (a) OPTICK_SAVE_CAPTURE(a)

#define PROFILE_EDITOR(a) { OPTICK_EVENT(a)
#define PROFILE_EDITOR_STOP }
*/
void EScene::Render(const Fmatrix& camera)
{
    if (!valid())
        return;

    SceneOToolsSet object_tools;
    SceneMToolsSet scene_tools;

    {
        PROFILE_EDITOR("Render Update Render List")



        SceneToolsMapPairIt t_it = m_SceneTools.begin();
        SceneToolsMapPairIt t_end = m_SceneTools.end();
        for (; t_it != t_end; t_it++)
        if (t_it->second)
        {
            // before render
            //t_it->second->BeforeRender(); 
            // sort tools
            ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(t_it->second);
            if (mt)
                object_tools.insert(mt);
            scene_tools.insert(t_it->second);
        }
 
        UpdateRenderList(&object_tools);

        PROFILE_EDITOR_STOP
    }
 
    {
        PROFILE_EDITOR("Render Traverse")

            // priority #0
            // normal
        mapRenderObjects.traverseLR(object_Normal_0);
        RenderScene(scene_tools, 0, false);
        // alpha
        mapRenderObjects.traverseRL(object_StrictB2F_0);
        // RENDER_SCENE_TOOLS(0, true);
        RenderScene(scene_tools, 0, true);
        
        // priority #1
        // normal
        mapRenderObjects.traverseLR(object_Normal_1);
        //RENDER_SCENE_TOOLS(1, false);
        RenderScene(scene_tools, 1, false);
        
        // alpha
        mapRenderObjects.traverseRL(object_StrictB2F_1);
        //RENDER_SCENE_TOOLS(1, true);
        RenderScene(scene_tools, 1, true);

        // priority #2
        // normal
        mapRenderObjects.traverseLR(object_Normal_2);
        //RENDER_SCENE_TOOLS(2, false);
        RenderScene(scene_tools, 2, false);

        // alpha
        mapRenderObjects.traverseRL(object_StrictB2F_2);
        //RENDER_SCENE_TOOLS(2, true);
        RenderScene(scene_tools, 2, true);

        // priority #3

        // normal
        mapRenderObjects.traverseLR(object_Normal_3);
        //RENDER_SCENE_TOOLS(3, false);
        RenderScene(scene_tools, 3, false);

        // alpha
        mapRenderObjects.traverseRL(object_StrictB2F_3);
        //RENDER_SCENE_TOOLS(3, true);
        RenderScene(scene_tools, 3, true);

        // render snap
        RenderSnapList();


        SceneMToolsIt s_it = scene_tools.begin();
        SceneMToolsIt s_end = scene_tools.end();
        for (; s_it != s_end; s_it++)
            (*s_it)->AfterRender();

        PROFILE_EDITOR_STOP
    }
 
    OPTICK_EVENT("");

}
//------------------------------------------------------------------------------

 
