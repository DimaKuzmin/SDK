#pragma once

#include <atomic>
#include "xrPhase_MergeLM_Rect.h"
#include "../XrLCLight/xrDeflector.h"
#include <mutex> 
#include <atomic>

class lm_layer;

class SurfacePlacePerpixel
{
public:
 	// Original 
	std::mutex csLMMerge;
	u32		alpha_ref = 254 - BORDER;
	u32		SurfaceGrid = 4096;
	u32		BORDER = 1;

	xr_vector<u8> surface_tbb;
	xr_vector<u16> occupied_y;

	u32 FilledCount = 0;
	u32 RegisterSize = 0;
  
	float GetMaxFilled();
 	void _InitSurface();
	bool _rect_register(L_rect& R, lm_layer* D);
	bool Place_Perpixel(L_rect& R, lm_layer* D);
	bool rect_place_full(L_rect& r, lm_layer* D);

	float FilledSize_cnt() { return float(FilledCount) / float(SurfaceGrid * SurfaceGrid); };
};
 
extern SurfacePlacePerpixel	placer_perpixel;
