#include "stdafx.h"
#include "xrLight_ImplicitDeflector.h"
#include "xrFace.h"
#include "xrLC_GlobalData.h"

ImplicitCalcGlobs cl_globs;
 
/** MAIN THREAD CALL EXECUTION, SORTING, SAVE**/
#include <ppl.h>
#include <atomic>
#include "../XrCDB/xrCDB.h"
#include "light_point.h"
#include "xrdeflector.h"

void ProcessingCPU()
{
	static std::atomic <int> Processed;
	Processed = 0;

	concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [](size_t threadID)
		{
			ImplicitDeflector& defl = cl_globs.DATA();
			CDB::COLLIDER DB; DB.ray_options(0);

			Fvector2 dim;   dim.set(float(defl.Width()), float(defl.Height()));
			Fvector2 half;  half.set(.5f / dim.x, .5f / dim.y);

			// Jitter data
			Fvector2 JS; JS.set(.499f / dim.x, .499f / dim.y);
			Fvector2* Jitter = nullptr; u32 Jcount = 0;
			Jitter_Select(Jitter, Jcount);

			while (true)
			{
				u32 V = Processed.fetch_add(1);
				AditionalData("Processed: %u/%u", Processed.load(), defl.Height());
				for (u32 U = 0; U < defl.Width(); U++)
				{
					base_color_c	C;
					u32	Fcount = 0;
					for (u32 J = 0; J < Jcount; J++)
					{
						// LUMEL space
						Fvector2				P;
						P.x = float(U) / dim.x + half.x + Jitter[J].x * JS.x;
						P.y = float(V) / dim.y + half.y + Jitter[J].y * JS.y;

						Fvector wP, wN, B;
						for (auto F : cl_globs.Hash().query(P.x, P.y))
						{
							_TCF& tc = F->tc[0];
							if (tc.isInside(P, B))
							{
								// We found triangle and have barycentric coords
								FromBarry(F, wP, wN, B);
								LightPoint(&DB, inlc_global_data()->RCAST_Model(), C, wP, wN, inlc_global_data()->L_static(), GetCurrentFlags(), F);
								Fcount++;
							}
						}
					}

					if (Fcount)
					{
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
			}
		});

}

void ImplicitLightingExec()
{
	if (g_params().m_quality == ebqDraft)		return;

 	xr_map<u32, ImplicitDeflector>		calculator;
 	cl_globs.Allocate();
 
	// Sorting
	Status("Sorting faces...");
	for (auto F : inlc_global_data()->g_faces())
	{
 		if (F->pDeflector)				continue;
		if (!F->hasImplicitLighting())	continue;
		
 		b_material&		M	= inlc_global_data()->materials()[F->dwMaterial];
		u32				Tid = M.surfidx;
		b_BuildTexture*	T	= &(inlc_global_data()->textures()[Tid]);
 
		auto		it	= calculator.find(Tid);
		if (it==calculator.end()) 
		{
			ImplicitDeflector	ImpD;
			ImpD.texture		= T;
			ImpD.faces.push_back(F);
			calculator.insert	(mk_pair(Tid,ImpD));
 		} 
		else 
		{
			ImplicitDeflector&	ImpD = it->second;
			ImpD.faces.push_back(F);
		}
	}

	// Lighing
	for (auto imp=calculator.begin(); imp!=calculator.end(); imp++)
	{
		ImplicitDeflector& defl = imp->second;
		Status			("Lighting implicit map '%s'...",defl.texture->name);
		Progress		(0);
		defl.Allocate	();
				
		// Setup cache
		Progress					(0);
		cl_globs.Initialize( defl );
 		
		//se7kills PPL Style MT
		if (gCompilerMode.CUDA)
		{
			extern void RunImplicitCuda();
			RunImplicitCuda();
		}
		else
		{
			ProcessingCPU();
		}

		defl.faces.clear();
		defl.SaveTexture();
 	}
 	
	cl_globs.Deallocate();
	calculator.clear	();	 
}