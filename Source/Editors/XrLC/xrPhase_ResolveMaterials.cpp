#include "stdafx.h"
#include "build.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"

extern void		Detach		(vecFace* S);

struct _counter
{
	u16	dwMaterial;
	u32	dwCount;
};


#include <mutex>
#include <execution>
#include <ppl.h>
#include <concurrent_vector.h>

std::mutex g_XSplit_mutex;

void	CBuild::xrPhase_ResolveMaterials()
{
	// Count number of materials
	CTimer t; t.Start();
	Status		("Calculating materials/subdivs... [%f]", t.GetElapsed_sec());
	// xr_vector<_counter>	counts;
	
	concurrency::concurrent_vector<_counter> counts_mt_safe;
	{
		counts_mt_safe.reserve		(256);
 		concurrency::parallel_for(size_t(0), size_t(lc_global_data()->g_faces().size()), [&](size_t Index)
		{
			BOOL	bCreate = TRUE;
			auto F = lc_global_data()->g_faces()[Index];
			for (u32 I = 0; I < counts_mt_safe.size(); I++)
			{
				if (F->dwMaterial == counts_mt_safe[I].dwMaterial)
				{
					counts_mt_safe[I].dwCount += 1;
					bCreate = FALSE;
					return;
 				}
			}

			if (bCreate)
			{
 				_counter	C;
				C.dwMaterial = F->dwMaterial;
				C.dwCount = 1;
 				counts_mt_safe.push_back(C);
			}
 		});
	}
	
	Status				("Perfroming subdivisions... [%f]", t.GetElapsed_sec());
	{
		concurrency::concurrent_vector<concurrency::concurrent_vector<Face*>> g_Xsplits_def;
		g_Xsplits_def.reserve(64 * 1024);
		g_Xsplits_def.resize(counts_mt_safe.size());

		concurrency::parallel_for_each(lc_global_data()->g_faces().begin(), lc_global_data()->g_faces().end(), [&](Face* F)
			{
				if (!F->Shader().flags.bRendering) return;

				for (u32 I = 0; I < counts_mt_safe.size(); I++)
				{
					if (F->dwMaterial == counts_mt_safe[I].dwMaterial)
					{
						g_Xsplits_def[I].push_back(F);
					}
				}
			});
 

		g_XSplit.reserve(64 * 1024);
		g_XSplit.resize(counts_mt_safe.size());
		for (auto i = 0; i < g_XSplit.size(); i++)
		{
			g_XSplit[i] = new vecFace( g_Xsplits_def[i].begin(), g_Xsplits_def[i].end() );
		}
	}

	Status				("Removing empty subdivs... [%f]", t.GetElapsed_sec());
	{
		for (int SP = 0; SP<int(g_XSplit.size()); SP++) 
		if (g_XSplit[SP]->empty())
			xr_delete(g_XSplit[SP]);
		g_XSplit.erase(std::remove(g_XSplit.begin(),g_XSplit.end(), (vecFace*) NULL), g_XSplit.end());
	}
	
	Status("Detaching subdivs... [%f]", t.GetElapsed_sec());
	{
		//for (u32 it = 0; it < g_XSplit.size(); it++)
		//	Detach(g_XSplit[it]);
			
		std::for_each( g_XSplit.begin(), g_XSplit.end(),
			[&](vecFace* F)
			{
				Detach(F);
			}
		);
	}


	clMsg				("%d subdivisions. total[%f]", g_XSplit.size(), t.GetElapsed_sec());
}
