#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrdeflector.h"
#include "..\LauncherSDL\xrThread.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrLightVertex.h"

#include "../../xrcore/xrSyncronize.h"
 
//#include "../xrLCLight/net_task_manager.h"
#include "../xrLCLight/mu_model_light.h"
#include "../XrLCLight/EmbreeRayTrace.h"

#include "../XrLCLight/base_face.h"
 
xrCriticalSection	task_CS;
xr_vector<int>		task_pool;
 
 
class CLMThread		: public CThread
{
private:
	HASH			H;
	CDB::COLLIDER	DB;
	base_lighting	LightsSelected;

public:
	CLMThread	(u32 ID) : CThread(ID)
	{
 		thMessages	= TRUE;
	}

	virtual void	Execute()
	{
 
		CDeflector* D	= 0;
 
		for (;;) 
		{
			// Get task
			task_CS.Enter		();
			
			thProgress			= 1.f - float(task_pool.size())/float(lc_global_data()->g_deflectors().size());
		
			if (task_pool.empty())	
			{
				task_CS.Leave		();
				return;
			}

			D					= lc_global_data()->g_deflectors()[task_pool.back()];

			 
			int IDX = lc_global_data()->g_deflectors().size() - task_pool.size();
			if (IDX % 512 == 0)
 			StatusNoMSG("DEFL[%d]/[%d], layer w[%d], h[%d]", 
				lc_global_data()->g_deflectors().size() - task_pool.size(), 
				lc_global_data()->g_deflectors().size(),
				D->layer.width, D->layer.height
			);
			 
			
			task_pool.pop_back	();
			task_CS.Leave		();
 
			// Perform operation
			try 
			{
				D->Light	(&DB,&LightsSelected,H);
			} 
			catch (...)
			{
				clMsg("* ERROR: CLMThread::Execute - light");
			}
		}
	}
};
 
#include "ppl.h"
#include <atomic>
  
void	CBuild::LMapsLocal				()
{
 	std::sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), [](const CDeflector* defl, const CDeflector* defl2)
	{
		return defl->similar_pos(*defl2, 0.1f);
	});
	 
	CTimer	start_time;
	start_time.Start();

	// Main process (4 threads) (-th MAX_THREADS)
	Status("Lighting...");
 
	CThreadManager	threads;
	for (u32 dit = 0; dit < lc_global_data()->g_deflectors().size(); dit++)
		task_pool.push_back(dit);

	for (int L = 0; L < gCompilerMode.ThreadsNum; L++)
		threads.start(xr_new<CLMThread>(L));
	threads.wait(500);
	 
	clMsg("%f seconds", start_time.GetElapsed_sec());
}

void	CBuild::LMaps					()
{
	//****************************************** Lmaps
	clMsg("Start Processing LMAPS: ");
	LMapsLocal();
}
 
void CBuild::RunMuModels()
{
 	//****************************************** Starting MU
   	mem_Compact();
	Light_prepare();
	//****************************************** Wait for MU
 	Phase("LIGHT: Waiting MU...");
 	wait_mu_base();
}


#include "xrLC.h"
 
void CBuild::Light()
{
	Msg("QUALYTI: %d, pixel: %f, jitter: %d", g_params().m_quality, g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);


	// Строим модель для Tracing
 	Phase("Building rcast-CFORM model...");
 	Light_prepare();
 	BuildRapid(TRUE);
	 
 
	//****************************************** Resolve materials
 	Phase("Resolving materials...");
 	xrPhase_ResolveMaterials();
	IsolateVertices(TRUE);

	//****************************************** UV mapping
 	Phase("Build UV mapping...");
 	xrPhase_UVmap();
	IsolateVertices(TRUE);

	//****************************************** Subdivide geometry
 	Phase("Subdividing geometry...");
 	xrPhase_Subdivide();

	//****************************************** Implicit

 	Phase("LIGHT: Implicit...");
	EmbreeMain.AttachGeometrys(true);
 	ImplicitLighting();

	Phase("LIGHT: LMaps...");
	EmbreeMain.AttachGeometrys(false);
 	LMaps();
	 
	//****************************************** Vertex
 	Phase("LIGHT: Vertex...");
 	LightVertex();

	//****************************************** Merge LMAPS
 	xrPhase_MergeLM();
  	xrPhase_SaveLmaps();

 	Phase("Merging geometry...");
 	xrPhase_MergeGeometry();
 	 

	EmbreeMain.AttachGeometrys(true);
	// Mu Models Lighting
  	RunMuModels();
    
	if (gCompilerMode.Embree)
		EmbreeMain.IntelEmbereUNLOAD();
}

void CBuild::LightVertex	()
{
	::LightVertex();
}