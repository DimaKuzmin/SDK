#include "stdafx.h"
#include "xrlight_implicitrun.h"

#include "..\LauncherSDL\xrThread.h"
#include "xrLight_Implicit.h"
#include "xrlight_implicitdeflector.h"
 
#include "tga.h"

#include "light_point.h"
#include "xrdeflector.h"
#include "xrLC_GlobalData.h"
#include "xrface.h"

#include "../../xrcdb/xrcdb.h"
#include <atomic>

std::atomic<int> curHeight;

xrCriticalSection csImplicit;
class ImplicitThread : public CThread
{
public:

 	ImplicitThread		(u32 ID, ImplicitDeflector* _DATA) : CThread (ID)
	{	
		thMessages = true;
	}


	/*** THREAD MAIN (ON CPU) ***/
	void ImplicitThread::Execute()
	{
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
		Sleep(0);

		thread_local CDB::COLLIDER DB;

		thread_local u32 Jcount;
		thread_local Fvector2* Jitter;
		thread_local Fvector2 dim;
		thread_local Fvector2 half;
		thread_local Fvector2 JS;

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


			if (V % 64 == 0 || V == defl.Height())
 				clMsg("$ CurV: %d, Sec[%.0f]", V, t.GetElapsed_sec());

			float progress = float(float(V) / float(defl.Height()));
 			ProgressMT(progress);
		}
	}
};


#include "ppl.h"
void PPL_MT()
{
	thread_local CDB::COLLIDER DB;

	thread_local u32 Jcount;
	thread_local Fvector2* Jitter;
	thread_local Fvector2 dim;
	thread_local Fvector2 half;
	thread_local Fvector2 JS;

	CTimer t;
	t.Start();

	std::atomic <int> Processed;

	concurrency::parallel_for(size_t(0), size_t(cl_globs.DATA().Height()), [&](size_t V)
		{

			ImplicitDeflector& defl = cl_globs.DATA();

			dim.set(float(defl.Width()), float(defl.Height()));
			half.set(.5f / dim.x, .5f / dim.y);

			// Jitter data
			JS.set(.499f / dim.x, .499f / dim.y);
			Jitter_Select(Jitter, Jcount);

			// Lighting itself
			DB.ray_options(0);

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

			if (Processed.load() % 64 == 0)
			{
				clMsg("Processed: %d, Timer: %f", Processed.load(), t.GetElapsed_sec());
			}
			Processed.fetch_add(1);
		}

	);

	ImplicitDeflector& defl = cl_globs.DATA();
	clMsg("Ligting Implicit: %s, Timer: %f", defl.texture->name, t.GetElapsed_sec());
}

 
void RunThread(ImplicitDeflector& defl)
{
	curHeight.store(0);

	CThreadManager			tmanager;
 
 	for (u32 thID = 0; thID < gCompilerMode.ThreadsNum; thID++)
	{
		ImplicitThread* th = xr_new<ImplicitThread>(thID, &defl);
 		tmanager.start(th);
	}

	tmanager.wait();
}	   

void RunImplicitMultithread(ImplicitDeflector& defl)
{
	// Start threads
	RunThread(defl);
}

