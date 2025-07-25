#pragma once

#include <atomic>
#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrdeflector.h" 
#include "../../XrCore/xrCore.h"

class lm_layer;

class SurfacePlacePerpixel
{
	
	u8*	 surface_tbb;
	u16* occupied_y = nullptr;
  
	// Initialization
 	u32 MAXPixelsCompression;

	// Filled size
	u32 StartYMin = 0;
	u32 FilledSize = 0;

	u32 SurfaceGrid = 0;
	xrCriticalSection csLMMerge;
	// Rendering of rect
	
public:
	u32 FilledPercent = 0;

	void RecalculateY();
	void _InitSurface_tbb();
	bool _rect_register_tbb(L_rect& R, lm_layer* D);
  
	bool Place_Perpixel_tbb(L_rect& R, lm_layer* D);
	bool rect_place_full(L_rect& r, lm_layer* D, u32 SizeX, u32 SizeY);
 
};
 
extern SurfacePlacePerpixel placer_perpixel;
 