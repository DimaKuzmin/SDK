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
   
void  object_Normal_0(EScene::mapObject_Node *N)	 
{
    if (N->val!=nullptr)
    (N->val)->RenderRoot(0, false);
}
void  object_Normal_1(EScene::mapObject_Node *N)	
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(1, false);
}
void  object_Normal_2(EScene::mapObject_Node *N)	 
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(2, false);
}
void  object_Normal_3(EScene::mapObject_Node *N)	 
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(3, false);
}

//------------------------------------------------------------------------------
void  object_StrictB2F_0(EScene::mapObject_Node *N)
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(0, true);
}
void  object_StrictB2F_1(EScene::mapObject_Node *N)
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(1, true);
}

void  object_StrictB2F_2(EScene::mapObject_Node *N)
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(2, true);
}

void  object_StrictB2F_3(EScene::mapObject_Node *N)
{
    if (N->val != nullptr)
    (N->val)->RenderRoot(3, true);
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

void RenderScene(SceneToolsMap& scene, int P, bool B)
{
     
    for (auto& tool : scene)
    {
        EDevice.SetShader(B ? EDevice.m_SelectionShader : EDevice.m_WireShader);
        RCache.set_xform_world(Fidentity);
        tool.second->OnRenderRoot(P, B);
    };
}

#include <ppl.h>
xrCriticalSection csEScene;
concurrency::task_group tasks;

int KeyCalc = 0;
int KeyRender = 1;


void EScene::UpdateRenderList(void* List, bool useMT)
{
    // extract and sort object tools
 
    tasks.wait();

    auto fun = [&]()
    {
        SceneOToolsSet object_tools;
        {
            OPTICK_FRAME("RenderList Update Thread");

            mapRenderObjects[KeyCalc].clear();

            object_tools.clear();

            SceneToolsMapPairIt t_it = m_SceneTools.begin();
            SceneToolsMapPairIt t_end = m_SceneTools.end();
            for (; t_it != t_end; t_it++)
                if (t_it->second)
                {
                    ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(t_it->second);
                    if (mt)
                        object_tools.insert(mt);
                }

            for (auto tool : object_tools)
            {
                ObjectList& list_objects = tool->GetObjects();

                for (auto O : list_objects)
                {
                    if (!O)
                        return;

                    if (O->Visible() && O->IsRender())
                    {
                        float distSQ = EDevice.vCameraPosition.distance_to_sqr(O->GetPosition());
                        mapRenderObjects[KeyCalc].insertInAnyWay(distSQ, O);
                    }
                }

            }
        }
    };

    if (useMT)
    {
        tasks.run(fun);
    }
    else
    {
        fun();
    }
};

extern bool NeedReupdate;
 
void EScene::Render(const Fmatrix& camera)
{
    if (!valid())
        return;

    int idx = KeyCalc;
    KeyRender = idx;
    KeyCalc   = (idx + 1) % 2;

    if (NeedReupdate)
    {
        KeyRender = 0;
        KeyCalc   = 0;
    }
    
    // EDevice.dwFrame % EDevice.RenderReloadObjectsTime == 0 
    {
        OPTICK_EVENT("Render Update Render List") 
        UpdateRenderList(0, !NeedReupdate);
    }
   
    auto& map = mapRenderObjects[KeyRender];

    if (map.size() == 0)
        return;

    {
        OPTICK_EVENT("Render Traverse")

         /*
        // priority #0
        // normal
        map.traverseLR(object_Normal_0);
        RenderScene(m_SceneTools, 0, false);
       
        // alpha
        map.traverseRL(object_StrictB2F_0);
        RenderScene(m_SceneTools, 0, true);
        
        // priority #1
        // normal
        map.traverseLR(object_Normal_1);
        RenderScene(m_SceneTools, 1, false);
        
        // alpha
        map.traverseRL(object_StrictB2F_1);
        RenderScene(m_SceneTools, 1, true);

        // priority #2
        // normal
        map.traverseLR(object_Normal_2);
        RenderScene(m_SceneTools, 2, false);

        // alpha
        map.traverseRL(object_StrictB2F_2);
        RenderScene(m_SceneTools, 2, true);

        // priority #3

        // normal
        map.traverseLR(object_Normal_3);
        RenderScene(m_SceneTools, 3, false);

        // alpha
        map.traverseRL(object_StrictB2F_3);
        RenderScene(m_SceneTools, 3, true);


         */
 
        if (m_SceneTools.empty())
            return;

        if (map.begin() == nullptr)
            return;
         
        // PRIORITY 0
        for (auto& O : map)
        {
             object_Normal_0(&O);
        }
        RenderScene(m_SceneTools, 0, false);
        
        for (auto& O : map)
        {
             object_StrictB2F_0(&O);
        }
        RenderScene(m_SceneTools, 0, true);
        
        // PRIORITY 1
        for (auto& O : map)
        {
             object_Normal_1(&O);
        }
        RenderScene(m_SceneTools, 1, false);
        
        for (auto& O : map)
        {
             object_StrictB2F_1(&O);
        }
        RenderScene(m_SceneTools, 1, true);
        
        
        // PRIORITY 2
        for (auto& O : map)
        {
             object_Normal_2(&O);
        }
        RenderScene(m_SceneTools, 2, false);
        
        for (auto& O : map)
        {
            if (&O)
            object_StrictB2F_2(&O);
        }
        RenderScene(m_SceneTools, 2, true);
        
        // PRIORITY 3
        for (auto& O : map)
        {
             object_Normal_3(&O);
        }
        RenderScene(m_SceneTools, 3, false);
        
        for (auto& O : map)
        {
             object_StrictB2F_3(&O);
        }
        RenderScene(m_SceneTools, 3, true);
 

        // render snap
        RenderSnapList();
 
        for (auto& tool : m_SceneTools)
            tool.second->AfterRender();
 
    }
  

}
//------------------------------------------------------------------------------

 
