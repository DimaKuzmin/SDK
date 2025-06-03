#include "stdafx.h" 
#include "xrPhase_MergeLM_Surface.h"

#include <cuda_runtime.h>
#include <ppl.h>
 
// #define USE_ACCELARATED
 

const	u32		alpha_ref = 254 - BORDER;
static __m256i mm256_alpha_ref = _mm256_set1_epi8(alpha_ref); // Замените alpha_value на нужное значение
static __m256i mm256_zero = _mm256_setzero_si256();
 
SurfacePlacePerpixel placer_perpixel;
 
// Surfaces
void SurfacePlacePerpixel::RecalculateY()
{
	for (int _Y = 0; _Y < SurfaceGrid; _Y++)
	{
		if (occupied_y[_Y] > MAXPixelsCompression)
			continue;
		StartYMin = _Y;
		break;
	}

}
void SurfacePlacePerpixel::_InitSurface_tbb()
{
	SurfaceGrid = getLMSIZE();
	
	surface_tbb = xr_alloc<u8>(SurfaceGrid * SurfaceGrid);
	FillMemory(surface_tbb, SurfaceGrid * SurfaceGrid, 0);
	 
	occupied_y = xr_alloc<u16>(SurfaceGrid);
	FillMemory(occupied_y, SurfaceGrid, 0);

	StartYMin = 0;
}
 
bool SurfacePlacePerpixel::_rect_register_tbb(L_rect& R, lm_layer* D)
{
	csLMMerge.Enter();

	// Для Многопотока нужно убедиться что точно не занято
	
	bool isCanRegister = Place_Perpixel_tbb(R, D);
 	if (isCanRegister)
	{
		u8* lm = &*(D->marker.begin());
		u32		s_x = D->width + 2 * BORDER;
		u32		s_y = D->height + 2 * BORDER;

		// Normal (and fastest way)
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
	}
 	csLMMerge.Leave();

	return isCanRegister;
}

bool SurfacePlacePerpixel::Place_Perpixel_tbb(L_rect& R, lm_layer* D)
{
	u8* lm = &*(D->marker.begin());
	u32	s_x = D->width + 2 * BORDER;
	u32	s_y = D->height + 2 * BORDER;
 
 	// Normal
 	
	for (u32 y = 0; y < s_y; y++)
	{		
		int x = 0;
		BYTE* P = surface_tbb + (y + R.a.y) * SurfaceGrid + R.a.x;
		u8* S = lm + y * s_x;

#ifdef USE_ACCELARATED 
		if (s_x > 32) // accelerated AVX2
		{
			int step = 32;
			for (x = 0; x < s_x - step; x += step, P += step, S += step)
			{
				__m256i mm_reg_s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(S));
				__m256i mm_reg_p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(P));

				__m256i mm_max = _mm256_max_epu8(mm_reg_s, mm256_alpha_ref);
				__m256i mm_cmp = _mm256_cmpeq_epi8(mm_max, mm256_alpha_ref);
				__m256i mm_andn = _mm256_andnot_si256(mm_cmp, mm_reg_p);
				__m256i mm_sad = _mm256_sad_epu8(mm_andn, mm256_zero); // AVX2 не имеет _mm256_sad_epu8, будет объяснение ниже


				// Здесь нужно суммировать содержимое mm_sad
				__m128i sum_lo = _mm256_castsi256_si128(mm_sad);         // нижние 128 бит
				__m128i sum_hi = _mm256_extracti128_si256(mm_sad, 1);    // верхние 128 бит
				__m128i sum = _mm_add_epi64(sum_lo, sum_hi);

				// Проверка суммы
				if (_mm_extract_epi64(sum, 0) != 0 || _mm_extract_epi64(sum, 1) != 0)
					return false;
			}
		}
#endif 
  		// destination scan-line
 		for (; x < s_x; x++, P++, S++)
		{
			if ( (*P) && (*S >= alpha_ref) ) 
 				return false;
		}		
 	}

	// It's OK to place it
	return true;
}
 
bool SurfacePlacePerpixel::rect_place_full(L_rect& r, lm_layer* D, u32 SizeX, u32 SizeY)
{
	MAXPixelsCompression = SurfaceGrid * gCompilerMode.LC_lmaps_max_pixels;

 	// Current Y Pos
	for (int _Y = 0; _Y < SurfaceGrid - SizeY; _Y++)
	{
		if (occupied_y[_Y] > MAXPixelsCompression)
			continue;

		L_rect R;
		BYTE* temp_surf = surface_tbb + _Y * SurfaceGrid;
		for (int _X = 0; _X < SurfaceGrid - SizeX; _X++)
		{
			if (_X + SizeX >= SurfaceGrid) break;

			R.init(_X, _Y, _X + SizeX, _Y + SizeY);	
			if (Place_Perpixel_tbb(R, D)) // Предварительный поиск
			{
 				if ( _rect_register_tbb(R, D) ) // Повторная проверка и рега
				{
 					r.set(R);
 					return TRUE;
				}
 			}
		}
	}

	return FALSE;
}
  