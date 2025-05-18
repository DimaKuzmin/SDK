#include "stdafx.h"
#include "xrlight_implicitrun.h"
#include "xrThread.h"
#include "xrLight_Implicit.h"
#include "xrlight_implicitdeflector.h"
 
#include "tga.h"

#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"

#include "../../xrcdb/xrcdb.h"
 

class ImplicitThread : public CThread
{
public:

	ImplicitExecute		execute;
	ImplicitThread		(u32 ID, ImplicitDeflector* _DATA) : CThread (ID), execute( ID )
	{	
	}

	virtual void		Execute	();
};

void	ImplicitThread ::	Execute	()
{
	// Priority
	SetThreadPriority		(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	Sleep					(0);
  	execute.Execute();
}

#include <atomic>

std::atomic<int> curHeight;
 
void RunThread(ImplicitDeflector& defl)
{
	curHeight.store(0);

	CThreadManager			tmanager;
 
 	for (u32 thID = 0; thID < gCompilerMode.ThreadsNum; thID++)
	{
		ImplicitThread* th = xr_new<ImplicitThread>(thID, &defl);
 		tmanager.start(th, thID);
	}

	tmanager.wait();
}	   

void RunImplicitMultithread(ImplicitDeflector& defl)
{
	// Start threads
	RunThread(defl);
}


/*** THREAD MAIN (ON CPU) ***/
 
void ImplicitExecute::Execute()
{
	ImplicitDeflector& defl = cl_globs.DATA();

	dim.set(float(defl.Width()), float(defl.Height()));
	half.set(.5f / dim.x, .5f / dim.y);

	// Jitter data
	JS.set(.499f / dim.x, .499f / dim.y);

	CTimer t;
	t.Start();
	Jitter_Select(Jitter, Jcount);

	// Lighting itself
	DB.ray_options(0);
  	for (;;)
	{
		int V = curHeight.load();
 		curHeight.fetch_add(1);
		if (V >= defl.Height())
			break;
		 
		// FOR CYCLE
		for (u32 U = 0; U < defl.Width(); U++)
		{
			base_color_c	C;

			u32				Fcount = 0;

			try
			{
				for (u32 J = 0; J < Jcount; J++)
				{
					// LUMEL space
					Fvector2				P;
					P.x = float(U) / dim.x + half.x + Jitter[J].x * JS.x;
					P.y = float(V) / dim.y + half.y + Jitter[J].y * JS.y;
					xr_vector<Face*>& space = cl_globs.Hash().query(P.x, P.y);

					// World space
					Fvector wP, wN, B;

					for (vecFaceIt it = space.begin(); it != space.end(); it++)
					{
						Face* F = *it;
						_TCF& tc = F->tc[0];
						if (tc.isInside(P, B))
						{
							// We found triangle and have barycentric coords
							Vertex* V1 = F->v[0];
							Vertex* V2 = F->v[1];
							Vertex* V3 = F->v[2];
							wP.from_bary(V1->P, V2->P, V3->P, B);
							wN.from_bary(V1->N, V2->N, V3->N, B);
							wN.normalize();
							u32 flags = (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0);
							LightPoint(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN, inlc_global_data()->L_static(), flags, F);

							// LightPointEmbree( C,wP, wN, inlc_global_data()->L_static(), flags, F);

							Fcount++;
						}
					}
				}
			}
			catch (...)
			{
				clMsg("* THREAD #%d: Access violation. Possibly recovered.");
			}

			if (Fcount)
			{
				// Calculate lighting amount
				C.scale(Fcount);
				C.mul(.5f);

				defl.Lumel(U, V)._set(C);
				defl.Marker(U, V) = 255;
			}
			else
			{
				defl.Marker(U, V) = 0;
			}

		}
		// FOR CYCLE END

		if (V % 128 == 0 || V == defl.Height())
		{
			clMsg("CurV: %d, Sec[%.0f]", V, t.GetElapsed_sec());
		}
	}
}