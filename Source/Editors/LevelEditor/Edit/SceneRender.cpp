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
   
void  object_Normal_0(EScene::RenderData *N)	 
{
    if (N->O!=nullptr)
        (N->O)->RenderRoot(0, false);
}

void  object_Normal_1(EScene::RenderData *N)
{
    if (N->O != nullptr)
       (N->O)->RenderRoot(1, false);
}
void  object_Normal_2(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(2, false);
}
void  object_Normal_3(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(3, false);
}

//------------------------------------------------------------------------------
void  object_StrictB2F_0(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(0, true);
}
void  object_StrictB2F_1(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(1, true);
}

void  object_StrictB2F_2(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(2, true);
}

void  object_StrictB2F_3(EScene::RenderData *N)
{
    if (N->O != nullptr)
        (N->O)->RenderRoot(3, true);
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
#include <execution>

void RenderScene(SceneToolsMap& scene, int P, bool B)
{
    for (auto& tool : scene)
    {
        EDevice.SetShader(B ? EDevice.m_SelectionShader : EDevice.m_WireShader);
        RCache.set_xform_world(Fidentity);
        tool.second->OnRenderRoot(P, B);
    };
}

concurrency::task_group tasks;

int KeyCalc = 0;
int KeyRender = 1;
 
 
void EScene::UpdateRenderList(void* List, bool useMT)
{
    /*
    // extract and sort object tools
    tasks.wait();

    int idx = KeyCalc;
    KeyRender = idx;
    KeyCalc = (idx + 1) % 2;
 
    auto task = [&]()
    {
        SceneOToolsSet object_tools;

        OPTICK_FRAME("RenderList Update Thread");
        OPTICK_EVENT("Render Update Render List")

        auto& map = mapRenderObjects[KeyCalc];
        map.clear(); // 

        for (auto& T : m_SceneTools)
        {
            ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(T.second);
            if (mt == nullptr)  continue;
        
            auto& Objects = mt->GetObjects();

            // if (EDevice.RenderTasks > 1)
            // {
            //     std::for_each(std::execution::par, Objects.begin(), Objects.end(), [&](CCustomObject* O)
            //     {
            //         if (O && O->Visible() && O->IsRender())
            //         {
            //             float distSQ = EDevice.vCameraPosition.distance_to_sqr(O->GetPosition());
            //             O->useInRender = true;
            //             O->distSQ = distSQ;
            // 
            // 
            //         }
            //         else
            //         {
            //             O->useInRender = false;
            //         }
            //     });
            // }
            // else
            // {
            //     for (auto O : Objects)
            //     {
            //         if (O && O->Visible() && O->IsRender())
            //         {
            //             float distSQ = EDevice.vCameraPosition.distance_to_sqr(O->GetPosition());
            //             O->useInRender = true;
            //             O->distSQ = distSQ;
            //         }
            //         else
            //         {
            //             O->useInRender = false;
            //         }
            //     };
            // }
             
            // OPTICK_EVENT("Render Set List")
            // for (auto O : Objects)
            // {
            //     if (O && O->useInRender)
            //     {
            //         RenderData data;
            //         data.distSQ = O->distSQ;
            //         data.O = O;
            //         map.push_back(data);
            //     }
            // }
        }

        

        // std::for_each(std::execution::par, threads_working.begin(), threads_working.end(), [&] (std::vector<CCustomObject*>& objects)
        // {
        //     std::vector<RenderData> datavec;
        //     for (auto O : objects)
        //     {
        //         if (O && O->Visible() && O->IsRender())
        //         {
        //             float distSQ = EDevice.vCameraPosition.distance_to_sqr(O->GetPosition());
        // 
        //             RenderData data;
        //             data.distSQ = distSQ;
        //             data.O = O;
        //             datavec.push_back(data);
        //         }
        //     }
        // 
        //     csRenderUpdate.Enter();
        //     map.resize(datavec.size());
        //     std::copy(datavec.begin(), datavec.end(), map.data());
        //     csRenderUpdate.Leave();
        // });
    };

    tasks.run(task);
    */
};


void EScene::RenderClearObjects()
{
    mapRenderObjects[0].clear();
    // mapRenderObjects[1].clear();
}
 
 
void EScene::Render(const Fmatrix& camera)
{
    if (!valid())
          return;

    auto& map = mapRenderObjects[0]; // KeyRender
    

    if (EDevice.dwFrame % EDevice.RenderReloadObjectsTime == 0)
    {
        map.clear();

        OPTICK_EVENT("Render Set List")
        // UpdateRenderList(0, true);

        xr_vector<CCustomObject*> objects;
    
        for (auto& T : Scene->m_SceneTools)
        {
            ESceneCustomOTool* mt = dynamic_cast<ESceneCustomOTool*>(T.second);
            if (mt == nullptr)
                continue;
     
            auto& Objects = mt->GetObjects();

            std::for_each(std::execution::par, Objects.begin(), Objects.end(), [&](CCustomObject* O)
            {
                if (O && O->Visible() && O->IsRender())
                {
                    float distSQ = EDevice.vCameraPosition.distance_to_sqr(O->GetPosition());
                    O->useInRender = true;
                    O->distSQ = distSQ;
                }
                else
                {
                    if (O != nullptr)
                        O->useInRender = false;
                }
            });

            for (auto O : Objects)
            {
                if (O && O->useInRender)
                {
                    RenderData data;
                    data.distSQ = O->distSQ;
                    data.O = O;
                    map.push_back(data);
                }
            }
             
        }       
    }

    if (map.size() == 0)
        return;

    {
        OPTICK_EVENT("Render Traverse") 
        if (m_SceneTools.empty())
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
