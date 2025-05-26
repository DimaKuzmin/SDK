#include "stdafx.h" 
#include "xrPhase_MergeLM_Surface.h"

#include <cuda_runtime.h>
 

void SurfacePlacePerpixel::_InitSurface_tbb()
{
	SurfaceGrid = getLMSIZE();
	surface_tbb = xr_alloc<u8>(SurfaceGrid * SurfaceGrid);
	FillMemory(surface_tbb, SurfaceGrid * SurfaceGrid, 0);

	surface_row_locks = xr_alloc<std::atomic<bool>>(SurfaceGrid);
	for (u32 i = 0; i < SurfaceGrid; ++i)
		surface_row_locks[i] = false; // unlocked

	occupied_y = xr_alloc<u16>(SurfaceGrid);
	FillMemory(occupied_y, SurfaceGrid, 0);
}

void SurfacePlacePerpixel::_rect_register_tbb(L_rect& R, lm_layer* D)
{
	u8* lm = &*(D->marker.begin());
	u32		s_x = D->width + 2 * BORDER;
	u32		s_y = D->height + 2 * BORDER;

	// Normal (and fastest way)
	csLMMerge.Enter();
	for (u32 y = 0; y < s_y; y++)
	{
		u32 _Y = y + R.a.y;

		BYTE* P = surface_tbb + _Y * SurfaceGrid + R.a.x;	// destination scan-line
		u8* S = lm + y * s_x;
		for (u32 x = 0; x < s_x; x++, P++, S++)
		{
			if (*S >= alpha_ref)
			{
				*P = 255;
				occupied_y[_Y] += 1;
			}
		}
	}
	csLMMerge.Leave();
}
 
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cuda.lib")
extern "C" cudaError_t cuda_place(u32 SurfaceGrid, u32 RectX, u32 RectY, u32 s_x, u32 s_y, u8* surface, u8* lightmap, bool& isFineded);
 
bool SurfacePlacePerpixel::CudaPlace(L_rect& R, lm_layer* D)
{
 	u8* lightmap = &*(D->marker.begin());

	bool isFineded = false;
	u32 RectX = R.a.x;
	u32 RectY = R.a.y;
	u32 s_x = D->width + 2 * BORDER;
	u32 s_y = D->height + 2 * BORDER;
	
	CTimer t; t.Start();
 	cuda_place(SurfaceGrid, RectX, RectY, s_x, s_y, surface_tbb, lightmap, isFineded);
 	Msg("(GPU) SX(%u) SZ(%u) Has Any Place : %u | ticks: %llu ", s_x, s_y, isFineded, t.GetElapsed_ticks());


	t.Start();
	isFineded = true;
	for (u32 y = 0; y < s_y; y++)
	{
		BYTE* P = surface_tbb + (y + R.a.y) * SurfaceGrid + R.a.x;	// destination scan-line
		u8* S = lightmap + y * s_x;
		for (u32 x = 0; x < s_x; x++, P++, S++)
		{
			if ((*P) && (*S >= alpha_ref))
				isFineded = false;
		}
	}
  	Msg("(CPU) SX(%u) SZ(%u) Has Any Place : %u | ticks: %llu ", s_x, s_y, isFineded, t.GetElapsed_ticks());

	return isFineded;
}

bool SurfacePlacePerpixel::Place_Perpixel_tbb(L_rect& R, lm_layer* D)
{
	u8* lm = &*(D->marker.begin());
	u32	s_x = D->width + 2 * BORDER;
	u32	s_y = D->height + 2 * BORDER;
	// Normal
 
	for (u32 y = 0; y < s_y; y++)
	{
  		BYTE* P = surface_tbb + (y + R.a.y) * SurfaceGrid + R.a.x;	// destination scan-line
		u8* S = lm + y * s_x;
		for (u32 x = 0; x < s_x; x++, P++, S++)
		{
			if ((*P) && (*S >= alpha_ref))
				return false;
		}
 	}
	 
	// It's OK to place it
	return true;
}



// Surfaces
bool SurfacePlacePerpixel::rect_place_full(L_rect& r, lm_layer* D, bool single_core)
{
	int SizeX = r.b.x;
	int SizeY = r.b.y;

	int x_max = SurfaceGrid - SizeX;
	int y_max = SurfaceGrid - SizeY;

	int y_max_line = SurfaceGrid * 0.72;

	auto set_surface_closed = [&](int start, int end, bool block)
		{
			if (!single_core)
				for (; start < end; start++)
				{
					if (block)
						surface_row_locks[start].store(true);
					else
						surface_row_locks[start].store(false);
				}
		};

	L_rect R;
	for (int _Y = 0; _Y < y_max; _Y++)
	{
		if (occupied_y[_Y] > y_max_line) // Нет Места под заливку
			continue;

		if (occupied_y[_Y] > SurfaceGrid - SizeX) // Нет Места под заливку
			continue;

		if (surface_row_locks[_Y].load()) // Заблокировано для мт
			continue;

		BYTE* temp_surf = surface_tbb + _Y * SurfaceGrid;
		set_surface_closed(_Y, _Y + SizeY, true);

		// remainder part
		for (int _X = 0; _X < x_max; _X++)
		{
			R.init(_X, _Y, _X + SizeX, _Y + SizeY);
			if (Place_Perpixel_tbb(R, D))
			{
				_rect_register_tbb(R, D);
				set_surface_closed(_Y, _Y + SizeY, false);
				r.set(R);
				return TRUE;
			}
		}

		set_surface_closed(_Y, _Y + SizeY, false);
	}
	return FALSE;
}

SurfacePlacePerpixel placer_perpixel;

 
