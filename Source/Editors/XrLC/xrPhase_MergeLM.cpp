#include "StdAfx.h"
#include "Build.h"
 
#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrDeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/Lightmap.h"

#define OFFSET_POS 4
  
void MergeLmap(vecDefl& Layer, CLightmap* lmap, int& MERGED)
{
	// Process 	
	// Немного отступ делаем
	u32 BorderUpdate = ((2 * BORDER) + OFFSET_POS);

	int _X = BorderUpdate, _Y = BorderUpdate;
	int _Max_y = 0;

	for (int it = 0; it < Layer.size(); it++)
	{
 		AditionalData("Process Y[%u] [%d]...Merged{%d}", _Y, it, MERGED);

		if (_Y > getLMSIZE() - 32) break;

		lm_layer& L = Layer[it]->layer;

		u32 WIDTH = L.width + (2 * BORDER);
		u32 HEIGHT = L.height + (2 * BORDER);

		if (_Max_y < HEIGHT)
			_Max_y = HEIGHT;

		if (_X + WIDTH > getLMSIZE() - 32)
		{
			_X = BorderUpdate;
			_Y += _Max_y + BorderUpdate;
			_Max_y = 0;
		}

		L_rect		rT, rS;
		rS.a.set(_X, _Y);
		rS.b.set(_X + WIDTH, _Y + HEIGHT);
		rS.iArea = L.Area();
		rT = rS;

		// Нужен только в оригенальной LMerge
		BOOL		bRotated = false;
		if (_Y < getLMSIZE() - HEIGHT)
		{
			lmap->Capture(Layer[it], rT.a.x, rT.a.y, rT.SizeX(), rT.SizeY(), bRotated);
			Layer[it]->bMerged = TRUE;
			MERGED++;
		}

		_X += WIDTH + BorderUpdate;
		Progress(float(it) / float(g_XSplit.size()));
	}
}


void CBuild::xrPhase_MergeLM_fast()
{
	Phase("Merging LMaps...");

	vecDefl			Layer;

	// **** Select all deflectors, which contain this light-layer
	Layer.clear	();
	for (u32 it=0; it<lc_global_data()->g_deflectors().size(); it++)
	{
		CDeflector*	D		= lc_global_data()->g_deflectors()[it];
		if (D->bMerged)		continue;
		Layer.push_back		(D);
	}
  
	// Merge this layer (which left unmerged)
	u32 StartSize = Layer.size();
	u32 TotalMerged = 0;
	
	string512	phase_name;
	xr_sprintf(phase_name, "Building lightmaps...");
	Phase(phase_name);

	CTimer t;
	while (Layer.size())
	{
		VERIFY( lc_global_data() );
 
		// Sort layer by similarity (state changes)
		// + calc material area
		Status		("Selection...");
		for (u32 it=0; it<materials().size(); it++) materials()[it].internal_max_area	= 0;
		for (u32 it=0; it<Layer.size(); it++)	
		{
			CDeflector*	D		= Layer[it];
			materials()[D->GetBaseMaterial()].internal_max_area	= _max(D->layer.Area(),materials()[D->GetBaseMaterial()].internal_max_area);
		}

		std::sort(Layer.begin(), Layer.end(),
			[](CDeflector* D1, CDeflector* D2) {
				if (D1->layer.height < D2->layer.height) {
					return true;
				} else {
					return false;
				}
			}
		);

		// Startup
		Status		("Processing...");
 
		CLightmap*	lmap		= new CLightmap ();
		VERIFY( lc_global_data() );
		lc_global_data()->lightmaps().push_back	(lmap);

  		int MERGED = 0;
		MergeLmap(Layer, lmap, MERGED);
 		TotalMerged += MERGED;
  
		
		// Remove merged lightmaps
		vecDeflIt last = std::remove_if(Layer.begin(), Layer.end(), [&](CDeflector* D) 
			{
				if (D->bMerged)
					return true;
				else
					return false;
			});

		Layer.erase(last, Layer.end());
		clMsg("Erase Layer(Deflects) Time: %u ms", t.GetElapsed_ms()); t.Start();
		
		// Save
  		AditionalData("Lmaps: %u, Merging:[%u/%u]|%u", lc_global_data()->lightmaps().size(), MERGED, TotalMerged, Layer.size());
 		Progress(float(float(MERGED) / float(StartSize)));
	}

	VERIFY(lc_global_data());
	clMsg("%d lightmaps builded", lc_global_data()->lightmaps().size());
	Progress(1.f);
}
    
void CBuild::xrPhase_SaveLmaps()
{
	Phase("Saving Light Maps...");
	Status("Destroying deflectors...");
	for (u32 it = 0; it < lc_global_data()->g_deflectors().size(); it++)
		xr_delete(lc_global_data()->g_deflectors()[it]);
	lc_global_data()->g_deflectors().clear();

	Status("Start Saving Lmaps: ");
	size_t USED_MEMORY = 0;

	CTimer t;
	int IDX = 0;
	for (auto lmap : lc_global_data()->lightmaps())
	{
		t.Start();
		lmap->Save(pBuild->path);
		clMsg("Saving Map [%u/%u] %u ms", IDX, lc_global_data()->lightmaps().size(), t.GetElapsed_ms());
		IDX++;
	}
}

