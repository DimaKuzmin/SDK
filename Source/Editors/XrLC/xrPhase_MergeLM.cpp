
#include "stdafx.h"
#include "build.h"

#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrlc_globaldata.h"
#include "../xrLCLight/lightmap.h"
// Surface access
extern void _InitSurface	();
extern BOOL _rect_place		(L_rect &r, lm_layer*		D);

IC int	compare_defl		(CDeflector* D1, CDeflector* D2)
{
	// First  - by material
	u16 M1		= D1->GetBaseMaterial();
	u16 M2		= D2->GetBaseMaterial();
	if (M1<M2)	return	1;  // less
	if (M1>M2)	return	0;	// more
	return				2;	// equal
}

// should define LESS(D1<D2) behaviour
// sorting - in increasing order
IC int	sort_defl_analyze	(CDeflector* D1, CDeflector* D2)
{
	// first  - get material index
	u16 M1		= D1->GetBaseMaterial();
	u16 M2		= D2->GetBaseMaterial();

	// 1. material area
	u32	 A1		= pBuild->materials()[M1].internal_max_area;
	u32	 A2		= pBuild->materials()[M2].internal_max_area;
	if (A1<A2)	return	2;	// A2 better
	if (A1>A2)	return	1;	// A1 better

	// 2. material sector (geom - locality)
	u32	 s1		= pBuild->materials()[M1].sector;
	u32	 s2		= pBuild->materials()[M2].sector;
	if (s1<s2)	return	2;	// s2 better
	if (s1>s2)	return	1;	// s1 better

	// 3. just material index
	if (M1<M2)	return	2;	// s2 better
	if (M1>M2)	return	1;	// s1 better

	// 4. deflector area
	u32 da1		= D1->layer.Area();
	u32 da2		= D2->layer.Area();
	if (da1<da2)return	2;	// s2 better
	if (da1>da2)return	1;	// s1 better

	// 5. they are EQUAL
	return				0;	// equal
}

// should define LESS(D1<D2) behaviour
// sorting - in increasing order
IC bool	sort_defl_complex	(CDeflector* D1, CDeflector* D2)
{
	if (D1->layer.height < D2->layer.height)
		return true;

	switch (sort_defl_analyze(D1,D2))	
	{
	case 1:		return true;	// 1st is better 
	case 2:		return false;	// 2nd is better
	case 0:		return false;	// none is better
	default:	return false;
	}
}

IC bool	sort_defl_fast(CDeflector* D1, CDeflector* D2)
{
	if (D1->layer.height > D2->layer.height)
		return true;
	else 
		return false;
}

class	pred_remove { public: IC bool	operator() (CDeflector* D) { { if (0==D) return TRUE;}; if (D->bMerged) {D->bMerged=FALSE; return TRUE; } else return FALSE;  }; };
 
#define MAX_THREADS 8

#include <thread>
xrCriticalSection csLM;

int LM_AREA_USED = 0;

void StartThread(int TH, vecDefl Layer, CLightmap* lmap, int start, int end, int* MERGED)
{
	for (int it = start; it != end; it++)
	{
		csLM.Enter();
		/*
		if (LM_AREA_USED > (1024 * 1024 ))
		{
			csLM.Leave();
			break;
		}
		*/
	 

		if (it % 2048 == 0)
			Msg("State [%d]", it);

		csLM.Leave();
		
		/*
		if (it % 256 == 0)
		{
			csLM.Enter();
			Msg("TH[%d] defl %d", TH, it);
			csLM.Leave();
		}
		*/

		lm_layer& L = Layer[it]->layer;
		L_rect		rT, rS;
		rS.a.set(0, 0);
		rS.b.set(L.width + 2 * BORDER - 1, L.height + 2 * BORDER - 1);
		rS.iArea = L.Area();
		rT = rS;
 
		if (_rect_place(rT, &L))
		{
			csLM.Enter();
			lmap->Capture(Layer[it], rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), rT.SizeX() != rS.SizeX());
			Layer[it]->bMerged = TRUE;
			MERGED++;
			LM_AREA_USED += L.Area();
			csLM.Leave();
 		}

	}

	//Msg("End TH [%d]", TH);
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
	


	u32 size_layer = 0;
	u32 LayerID = 0;

	bool fastWay = strstr(Core.Params, "-fast_lightmaps");

	CTimer timer; timer.Start();
	// Merge this layer (which left unmerged)

	if (fastWay)
	{
		int area = 0;
 
		for (int it = 0; it < Layer.size(); it++)
		{
			lm_layer& L = Layer[it]->layer;
			area += L.Area();
		}

		int lm_1024 = 1024 * 1024;
		int lm_2048 = 2048 * 2048;
		int lm_4096 = 4096 * 4096;
		int lm_8192 = 8192 * 8192;

		int use_size = 1024;

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
		setLMSIZE(use_size);
	}
	 
	while (Layer.size()) 
	{
		LM_AREA_USED = 0;
 
		LayerID += 1;

		VERIFY( lc_global_data() );
		string512	phase_name;
		xr_sprintf		(phase_name,"Building lightmap %d ... Layers[%d]", lc_global_data()->lightmaps().size(), Layer.size());
		Phase		(phase_name);

		// Sort layer by similarity (state changes)
		// + calc material area
		Status		("Sort lightmaps size + Calc Material...");

		for (u32 it = 0; it < materials().size(); it++)
			materials()[it].internal_max_area = 0;

		for (u32 it=0; it<Layer.size(); it++)	
		{
			CDeflector*	D		= Layer[it];
			materials()[D->GetBaseMaterial()].internal_max_area	= _max(D->layer.Area(),materials()[D->GetBaseMaterial()].internal_max_area);
		}

		if (!fastWay)
		{
			std::stable_sort(Layer.begin(), Layer.end(), sort_defl_complex);
		}
		else
		{
			//SE7kills
			std::sort(Layer.begin(), Layer.end(), sort_defl_fast);
		}

		// Startup
		Status		("Processing...");
		_InitSurface			();
		CLightmap*	lmap		= xr_new<CLightmap> ();
		VERIFY( lc_global_data() );
		lc_global_data()->lightmaps().push_back	(lmap);
 
		int MERGED = 0;

		if (fastWay)
		{
			// Process 	
			int x = 0, y = 0;
			u16 prev_resize_height = 0;
			u16 prev_resize_width = 0;
			u16 max_y = 0;


			for (int it = 0; it < Layer.size(); it++)
			{
				lm_layer& L = Layer[it]->layer;
				if (max_y < L.height + 2)
					max_y = L.height + 2;

				if (x + L.width + 2 > getLMSIZE() - 16 - L.width)
				{
					x = 0;  y += max_y; max_y = 0;
				}

				{
					L_rect		rT, rS;
					rS.a.set(x, y);
					rS.b.set(x + L.width + 2 * BORDER - 1, y + L.height + 2 * BORDER - 1);
					rS.iArea = L.Area();
					rT = rS;

					x += L.width + 2;

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
			

		// Process 
		
		if (!fastWay)
		{
			u32 maxarea = getLMSIZE() * getLMSIZE() * 8;	// Max up to 8 lm selected
			u32 curarea = 0;
			u32 merge_count = 0;

			// Select first deflectors which can fit
			Status("Selection...");
			for (u32 it = 0; it < (int)Layer.size(); it++)
			{
				int		defl_area = Layer[it]->layer.Area();
				if (curarea + defl_area > maxarea) break;
				curarea += defl_area;
				merge_count++;
			}

			if (Layer.size() > 1024)
			{
				std::thread* th = new std::thread[MAX_THREADS];

				int split = merge_count / 8;

				for (int i = 0; i < 8; i++)
				{
					th[i] = std::thread(StartThread, i, Layer, lmap, i * split, split * (i + 1), &MERGED);
				}

				for (int i = 0; i < 8; i++)
				{
					th[i].join();
				}
			}
			else
			{
				for (u32 it = 0; it < merge_count; it++)
				{
					if (0 == (it % 1024))
						Status("Process [%d/%d]...", it, merge_count);

					lm_layer& L = Layer[it]->layer;
					L_rect		rT, rS;
					rS.a.set(0, 0);
					rS.b.set(L.width + 2 * BORDER - 1, L.height + 2 * BORDER - 1);
					rS.iArea = L.Area();
					rT = rS;
					if (_rect_place(rT, &L))
					{
						lmap->Capture(Layer[it], rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), rT.SizeX() != rS.SizeX());
						Layer[it]->bMerged = TRUE;
						MERGED++;
					}
					Progress(float(LM_AREA_USED / (1024 * 1024)));
					//Progress(float(it) / float(merge_count));
				}
			}
		}
   
		/*
		for (u32 it = 0; it < merge_count; it++)
		{
			if (0 == (it % 1024))	
				Status("Process [%d/%d]...", it, merge_count);
			lm_layer& L = Layer[it]->layer;
			L_rect		rT, rS;
			rS.a.set(0, 0);
			rS.b.set(L.width + 2 * BORDER - 1, L.height + 2 * BORDER - 1);
			rS.iArea = L.Area();
			rT = rS;
			if (_rect_place(rT, &L))
			{
 				lmap->Capture(Layer[it], rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), rT.SizeX() != rS.SizeX());
				Layer[it]->bMerged = TRUE;
				MERGED++;
			}
			Progress(float(it) / float(merge_count));
		}
		 */

		

 
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
		for (int i = 0; i < Layer.size();i++)
 			Msg("w: %d, h: %d", Layer[i]->layer.width, Layer[i]->layer.height);
 
 

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

 