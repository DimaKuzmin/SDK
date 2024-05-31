
#include "stdafx.h"
#include "build.h"

#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrlc_globaldata.h"
#include "../xrLCLight/lightmap.h"
// Surface access
IC bool	sort_defl_fast(CDeflector* D1, CDeflector* D2)
{
	if (D1->layer.height < D2->layer.height)
		return true;
	else 
		return false;
}

class	pred_remove { public: IC bool	operator() (CDeflector* D) { { if (0==D) return TRUE;}; if (D->bMerged) {D->bMerged=FALSE; return TRUE; } else return FALSE;  }; };
   
#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

void SelectionLmapSize(vecDefl& Layer)
{
	u64 area = 0;
	int lm_1024 = 1024 * 1024;
	int lm_2048 = 2048 * 2048;
	int lm_4096 = 4096 * 4096;
	int lm_8192 = 8192 * 8192;

	for (int it = 0; it < Layer.size(); it++)
	{
		if (lm_8192 < area)
			break;

		lm_layer& L = Layer[it]->layer;
		area += L.Area();
	}

	int use_size = 8192;

	if (area < lm_1024)
	{
		use_size = 1024;
	}
	else if (area < lm_2048)
	{
		use_size = 2048;
	}
	else if (area < lm_4096)
	{
		use_size = 4096;
	}
	else if (area < lm_8192)
	{
		use_size = 8192;
	}

	clMsg("Select LM_SIZE: %d", use_size);
	setLMSIZE(use_size);
}
 
void MergeLmap(vecDefl& Layer, CLightmap* lmap, int& MERGED)
{
	// Process 	
	int x = 0, y = 0;
	u16 prev_resize_height = 0;
	u16 prev_resize_width = 0;
	u16 max_y = 0;


	for (int it = 0; it < Layer.size(); it++)
	{
		lm_layer& L = Layer[it]->layer;
		if (max_y < L.height + 5)
			max_y = L.height + 5;

		if (x + L.width + 2 > getLMSIZE() - 16 - L.width)
		{
			x = 0;  y += max_y + 5; max_y = 0;
		}

		{
			L_rect		rT, rS;
			rS.a.set(x, y);
			rS.b.set(x + L.width + 2 * BORDER - 1, y + L.height + 2 * BORDER - 1);
			rS.iArea = L.Area();
			rT = rS;

			x += L.width + 5;

			BOOL		bRotated = rT.SizeX() != rS.SizeX();

			if (y < getLMSIZE() - 16 - L.height)
			{
				lmap->Capture(Layer[it], rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), bRotated);
				Layer[it]->bMerged = TRUE;
				MERGED++;
			}
		}

		Progress(float(it) / float(g_XSplit.size()));

		if (0 == (it % 1024))
			Status("Process [%d/%d]...", it, g_XSplit.size());
	}
}


void CBuild::xrPhase_MergeLM()
{
	vecDefl			Layer;

	// **** Select all deflectors, which contain this light-layer
	Layer.clear	();
	for (u32 it=0; it<lc_global_data()->g_deflectors().size(); it++)
	{
		CDeflector*	D		= lc_global_data()->g_deflectors()[it];
		if (D->bMerged)		continue;
		Layer.push_back		(D);
	}
	
	// Update LMAP DDS FILE (W*H)
	SelectionLmapSize(Layer);

	u32 size_layer = 0;
	u32 LayerID = 0;

	bool fastWay = true;  

	CTimer timer; timer.Start();
	// Merge this layer (which left unmerged)
 
	while (Layer.size()) 
	{ 
		LayerID += 1;

		VERIFY( lc_global_data() );
		string512	phase_name;
		xr_sprintf		(phase_name,"Building lightmap %d ... Layers[%d]", lc_global_data()->lightmaps().size(), Layer.size());
		Phase		(phase_name);

		// Sort layer by similarity (state changes)
		// + calc material area
		Status		("Sort lightmaps, Calc Material, Size DDS...");

		for (u32 it = 0; it < materials().size(); it++)
			materials()[it].internal_max_area = 0;

		for (u32 it=0; it<Layer.size(); it++)	
		{
			CDeflector*	D		= Layer[it];
			materials()[D->GetBaseMaterial()].internal_max_area	= _max(D->layer.Area(),materials()[D->GetBaseMaterial()].internal_max_area);
		}

 		//SE7kills
		std::sort(Layer.begin(), Layer.end(), sort_defl_fast);



		// Startup
		Status		("Processing...");
 
 		CLightmap*	lmap		= xr_new<CLightmap> ();
		VERIFY( lc_global_data() );
		lc_global_data()->lightmaps().push_back	(lmap);
 
		// Calculate X, Y Ñìåùåíèå
		int MERGED = 0;
 		MergeLmap(Layer, lmap, MERGED);
   
		clMsg("MERGED: %d, TOT: %d", MERGED, Layer.size());
		int recvest = Layer.size() - MERGED;

		Progress	(1.f);

		// Remove merged lightmaps
		Status			("Cleanup...");
		vecDeflIt last	= std::remove_if	(Layer.begin(), Layer.end(), pred_remove());
		Layer.erase		(last, Layer.end());

		clMsg("LM After Clean %d==%d", Layer.size(), recvest);


		if (lmap->lm.width != 0 && lmap->lm.height != 0)
		{
			// Save
			Status("Saving...");
			VERIFY(pBuild);
			lmap->Save(pBuild->path);
		}
		else
		{
			clMsg("CRITICAL ÍÅ×ÅÃÎ ÍÅÒ Â ËÀÉÒ ÌÀÏÅ");
		}
 	}

	clMsg("MERGE LIGHT MAPS %d ms", timer.GetElapsed_ms() );

	VERIFY( lc_global_data() );
	clMsg		( "%d lightmaps builded", lc_global_data()->lightmaps().size() );

	// Cleanup deflectors
	Progress	(1.f);
	Status		("Destroying deflectors...");
	for (u32 it=0; it<lc_global_data()->g_deflectors().size(); it++)
		xr_delete(lc_global_data()->g_deflectors()[it]);
	lc_global_data()->g_deflectors().clear_and_free	();
}

 