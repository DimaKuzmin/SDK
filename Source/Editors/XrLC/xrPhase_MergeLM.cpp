#include "StdAfx.h"
#include "Build.h"

#include "xrPhase_MergeLM_Surface.h"
#include "xrPhase_MergeLM_Rect.h"
#include "../XrLCLight/xrDeflector.h"
#include "../XrLCLight/xrLC_GlobalData.h"
#include "../XrLCLight/Lightmap.h"

#include <ppl.h>

extern CompilersMode gCompilerMode;
 
// Other Stuff
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

void CBuild::xrPhase_SortDeflectors()
{
	for (u32 it = 0; it < materials().size(); it++)
		materials()[it].internal_max_area = 0;

	for (auto D : lc_global_data()->g_deflectors())
		materials()[D->GetBaseMaterial()].internal_max_area = std::max(D->layer.Area(), materials()[D->GetBaseMaterial()].internal_max_area);

	std::stable_sort(lc_global_data()->g_deflectors().begin(), lc_global_data()->g_deflectors().end(), sort_defl_complex);
}

// Processing 
u32 MergeLmap_Compact(xr_vector<CDeflector*>& Layer, CLightmap* lmap)
{
	// Sort layer by similarity (state changes) + calc material area
	// Слишком много возьмет для помещения 
	static xrCriticalSection IndexLock;

	static u32 MergedCount = 0;
 	static std::atomic<u32> CurrentIndex = 0;
	static std::atomic<u32> ErrorsPlace = 0;

	CurrentIndex = 0;
	ErrorsPlace  = 0;
	MergedCount  = 0;
	concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [lmap, Layer](size_t thread_id)
	{
		u32 MaxSize = Layer.size();
 		while (true)
		{
			u32 IndexTask = CurrentIndex.fetch_add(1);
			if (IndexTask >= MaxSize) break;
			auto D = Layer[IndexTask];
			lm_layer& L = D->layer;
			 			
			if (ErrorsPlace.load() > 256 && L.Area() > 4) { continue; }; // нету места !
			if (ErrorsPlace.load() > 4096) break;	// Вообще не влезло !
  
			AditionalData("ID(%u/%u) filled(%f)", IndexTask, MaxSize, placer_perpixel.FilledSize_cnt());

			u8 BORDER = gCompilerMode.LC_lmap_BORDER;

			L_rect		rT, rS;
			rS.a.set(0, 0);
			rS.b.set(L.width + 2 * BORDER - 1, L.height + 2 * BORDER - 1);
			rS.iArea = L.Area();
			rT = rS;

 			if (placer_perpixel.rect_place_full(rT, &L))
			{
				IndexLock.Enter();
				if (D->bMerged == false)
				{
					lmap->Capture(D, rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), false);
					D->bMerged = true;
					D->layer.clear_memory();
					MergedCount++;
				}
				IndexLock.Leave();
			}
			else
				if (L.Area() < 128)
					ErrorsPlace.fetch_add(1);
		}
	});
	
	// Удаляем то что сделали !
	Layer.erase(std::remove_if(Layer.begin(), Layer.end(),
		[](CDeflector* D)
		{
			if (D == nullptr)return true;
			if (D->bMerged) return true;
			return false;
		}), Layer.end()
	);

	Progress(1.f);
	return MergedCount;
}

void CBuild::xrPhase_MergeLM(xr_vector<CDeflector*>& Layer)
{
 	Phase("Building Lmaps ...");
	xrPhase_SortDeflectors();

	auto& LM = lc_global_data()->lightmaps();
	// Merge this layer (which left unmerged)
	while (Layer.size())
	{
 		// Processing Big Vec	
		CLightmap* BuildingLmap = new CLightmap();
		LM.push_back(BuildingLmap);
		placer_perpixel._InitSurface();
		
   		MergeLmap_Compact(Layer, BuildingLmap);
 		
		u32 saving_dds = 0;
		BuildingLmap->Save(pBuild->path, saving_dds);
		
		clMsg("* [Lightmap: %u] : Saved Filled as(%f)", LM.size(), placer_perpixel.FilledSize_cnt() );
	}
}
 