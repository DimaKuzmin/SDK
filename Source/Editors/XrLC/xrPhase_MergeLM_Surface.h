#pragma once

#include <atomic>
#include "xrPhase_MergeLM_Rect.h"
#include "../xrLCLight/xrdeflector.h" 
#include "../../XrCore/xrCore.h"

class lm_layer;

class SurfacePlacePerpixel
{
	u8 * surface_tbb = nullptr;
	u16* occupied_y = nullptr;

	const	u32		alpha_ref = 254 - BORDER;
	std::atomic<bool>* surface_row_locks = nullptr;

	// Initialization
	u32 SurfaceGrid = 0;
	xrCriticalSection csLMMerge;
	// Rendering of rect
	
public:
	void _InitSurface_tbb();
	void _rect_register_tbb(L_rect& R, lm_layer* D);

	bool CudaPlace(L_rect& R, lm_layer* D);

	bool Place_Perpixel_tbb(L_rect& R, lm_layer* D);

	bool rect_place_full(L_rect& r, lm_layer* D, bool single_core);

};

extern SurfacePlacePerpixel placer_perpixel;