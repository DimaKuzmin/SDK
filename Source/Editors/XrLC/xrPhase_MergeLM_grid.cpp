#include "StdAfx.h"
#include "Build.h"

#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrDeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/Lightmap.h"
#include <atomic>
#include <ppl.h>

#include "xrPhase_MergeLM_Surface.h"

// Surface access
#define SCALE_SIZE  2

xrCriticalSection csMergeLM;

IC int	compare_defl(CDeflector* D1, CDeflector* D2)
{
	// First  - by material
	u16 M1 = D1->GetBaseMaterial();
	u16 M2 = D2->GetBaseMaterial();
	if (M1 < M2)	return	1;  // less
	if (M1 > M2)	return	0;	// more
	return				2;	// equal
}

// should define LESS(D1<D2) behaviour
// sorting - in increasing order
IC int	sort_defl_analyze(CDeflector* D1, CDeflector* D2)
{
	// first  - get material index
	u16 M1 = D1->GetBaseMaterial();
	u16 M2 = D2->GetBaseMaterial();

	// 1. material area
	u32	 A1 = pBuild->materials()[M1].internal_max_area;
	u32	 A2 = pBuild->materials()[M2].internal_max_area;
	if (A1 < A2)	return	2;	// A2 better
	if (A1 > A2)	return	1;	// A1 better

	// 2. material sector (geom - locality)
	u32	 s1 = pBuild->materials()[M1].sector;
	u32	 s2 = pBuild->materials()[M2].sector;
	if (s1 < s2)	return	2;	// s2 better
	if (s1 > s2)	return	1;	// s1 better

	// 3. just material index
	if (M1 < M2)	return	2;	// s2 better
	if (M1 > M2)	return	1;	// s1 better

	// 4. deflector area
	u32 da1 = D1->layer.Area();
	u32 da2 = D2->layer.Area();
	if (da1 < da2)return	2;	// s2 better
	if (da1 > da2)return	1;	// s1 better

	// 5. they are EQUAL
	return				0;	// equal
}

// should define LESS(D1<D2) behaviour
// sorting - in increasing order
IC bool	sort_defl_complex(CDeflector* D1, CDeflector* D2)
{
	switch (sort_defl_analyze(D1, D2))
	{
	case 1:		return true;	// 1st is better 
	case 2:		return false;	// 2nd is better
	case 0:		return false;	// none is better
	default:	return false;
	}
}

class	pred_remove 
{ 
	public: IC bool	operator() (CDeflector* D) 
	{ 
		if (0 == D)
			return TRUE; 
		
		if (D->bMerged)
		{
			D->bMerged = FALSE; 
			return TRUE; 
		} 
		else 
			return FALSE; 
	};
};

#pragma optimize("", off)

struct CapturedMap
{
	CDeflector* deflector;
	L_rect rect;
	u32 Index = 0;
};
 
void CBuild::xrPhase_MergeLM()
{
	if (gCompilerMode.LC_lmaps_alternative)
	{
		xrPhase_MergeLM_fast();
		return;
	}
 
	vecDefl			Layer;

	// **** Select all deflectors, which contain this light-layer
	Layer.clear();
	for (u32 it = 0; it < lc_global_data()->g_deflectors().size(); it++)
	{
		CDeflector* D = lc_global_data()->g_deflectors()[it];
		if (D->bMerged)		continue;
		Layer.push_back(D);
	}

	// Merge this layer (which left unmerged)
	while (Layer.size())
	{
		VERIFY(lc_global_data());
		string512	phase_name;
		sprintf(phase_name, "Building lightmap %d...", lc_global_data()->lightmaps().size());
		Phase(phase_name);

		// Sort layer by similarity (state changes)
		// + calc material area
		Status("Selection...");
		for (u32 it = 0; it < materials().size(); it++)
			materials()[it].internal_max_area = 0;

		for (u32 it = 0; it < Layer.size(); it++)
		{
			CDeflector* D = Layer[it];
			materials()[D->GetBaseMaterial()].internal_max_area = _max(D->layer.Area(), materials()[D->GetBaseMaterial()].internal_max_area);
		}

		std::stable_sort(Layer.begin(), Layer.end(), sort_defl_complex);

		// Select first deflectors which can fit

 		u32 maxarea		= getLMSIZE() * getLMSIZE() * SCALE_SIZE;	// Max up to 8 lm selected
		u32 curarea		= 0;
		u32 merge_count = 0;

		for (u32 it = 0; it < (int)Layer.size(); it++)
		{
			int		defl_area = Layer[it]->layer.Area();
			if (curarea + defl_area > maxarea) break;
			curarea += defl_area;
			merge_count++;
		}
 		
 		// Startup
		CTimer tStat;
		tStat.Start();

		Status("Processing...");
		CLightmap* lmap = new CLightmap();
		lc_global_data()->lightmaps().push_back(lmap);

		// Process 
		u32 MergedSize = 0;
		u32 CurrentIndex = 0;
   		xr_vector<CapturedMap> capture;  
 		std::atomic<u32> Errors = 0;
		
		// Calculate Rects
 		auto calculate_maps = [&]()
		{
			while (true)
			{
				csMergeLM.Enter();
				u32 it = CurrentIndex;
				CurrentIndex += 1;
				csMergeLM.Leave();
				 
				if (it >= merge_count || Errors.load() > 10000)	break;
  
 				CDeflector* deflector = Layer[it];
  				if (deflector->bMerged) continue;
				 
				lm_layer& L = deflector->layer;
				u32 SizeX = L.width + 2 * BORDER - 1;
				u32 SizeY = L.height + 2 * BORDER - 1;

				L_rect		rT, rS;
				rS.a.set(0, 0);
				rS.b.set(SizeX, SizeY);
				rS.iArea = L.Area();
				rT = rS;

				AditionalData("IT:%u/%u | merged:%u | errors: %u", it, merge_count, MergedSize, Errors.load());
	 
				if ( placer_perpixel.rect_place_full(rT, &L, SizeX, SizeY) )
				{
					csMergeLM.Enter();
					if (!deflector->bMerged)
					{
						lmap->Capture(deflector, rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), false);
						deflector->bMerged = true;
						MergedSize++;
					}
					csMergeLM.Leave();
 				}
				else
 					Errors.fetch_add(1);
   				ProgressMT(float(it) / float(merge_count));
			}
		};
 
		// Single Core To ALL Process
 
		placer_perpixel._InitSurface_tbb();
  		concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [&](size_t thread_id)
		{
			calculate_maps();
		});
 		
		// calculate_maps(true);
		clMsg("ProccLmap: %u | Merged: %u", lc_global_data()->lightmaps().size(), MergedSize);

 
 		Progress(1.f);

		// Remove merged lightmaps
		Status("Cleanup...");
		vecDeflIt last = std::remove_if(Layer.begin(), Layer.end(), pred_remove());
		Layer.erase(last, Layer.end());
	}
	VERIFY(lc_global_data());
	clMsg("%d lightmaps builded", lc_global_data()->lightmaps().size());
}
