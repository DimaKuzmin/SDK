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
#include <atomic>
#include "ppl.h" 

xrCriticalSection csEnterMUThread;
u32 atomic_u32 = 0;


class CLMThread : public CThread
{
private:
	HASH			H;
	CDB::COLLIDER	DB;
	base_lighting	LightsSelected;

public:
	CLMThread(u32 ID) : CThread(ID)
	{
		thMessages = TRUE;
	}

	virtual void	Execute()
	{

		CDeflector* D = 0;

		for (;;)
		{
			// Get task
			csEnterMUThread.Enter();
 			if (atomic_u32 >= lc_global_data()->g_deflectors().size()) 
			{
				csEnterMUThread.Leave();
				break;
			}

			D = lc_global_data()->g_deflectors()[atomic_u32];
 			AditionalData("Index [%d]/[%d], w[%d], h[%d]", atomic_u32, lc_global_data()->g_deflectors().size(), D->layer.width, D->layer.height );
 			atomic_u32++;
			csEnterMUThread.Leave();

			// Perform operation
			try
			{
				D->Light(&DB, &LightsSelected, H);
			}
			catch (...)
			{
				clMsg("* ERROR: CLMThread::Execute - light");
			}
		}
	}
};

void	CBuild::LMapsLocal()
{
//	std::sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), [](const CDeflector* defl, const CDeflector* defl2)
//	{
//		return defl->similar_pos(*defl2, 0.1f);
//	});

	CTimer	start_time;
	start_time.Start();

	// Main process (4 threads) (-th MAX_THREADS)
	Status("Lighting...");
	atomic_u32 = 0;
	CThreadManager	threads;
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

// #define FAST_RAYTRACE

void CBuild::Light()
{
	Msg("QUALYTI: %d, pixel: %f, jitter: %d", g_params().m_quality, g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);
	
#ifndef FAST_RAYTRACE
	Phase("Adaptive HT...");
	xrPhase_AdaptiveHT();
#endif

	Phase("Building normals...");
	CalcNormals();

#ifndef FAST_RAYTRACE
	Phase("Building collision database...");
	BuildCForm();
#endif
	 
	// Строим модель для Tracing
 	Phase("Building rcast-CFORM model...");
 	Light_prepare();
 	BuildRapid(TRUE, TRUE);
	  
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
 
 	// Phase("LIGHT: Implicit...");
	// EmbreeMain.AttachGeometrys(true);
 	// ImplicitLighting();

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