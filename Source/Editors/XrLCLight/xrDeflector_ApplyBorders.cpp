#include "stdafx.h"
#include "xrDeflector.h"
 
// RMS  TESTING 
bool compress_Zero(lm_layer& lm, u32 rms)
{
	auto rms_average = [](lm_layer& lm, base_color_c& C)
	{
		u32 x, y, _count = 0;
		for (y = 0; y < lm.height; y++)
		{
			for (x = 0; x < lm.width; x++)
			{
				u32	offset = y * lm.width + x;
				if (lm.marker[offset] >= 254)
				{
					base_color_c	cc;
					lm.surface[offset]._get(cc);
					C.add(cc);
					_count++;
				}
			}
		}
		return	_count;
	};

	auto rms_test = [](lm_layer& lm, u32 _r, u32 _g, u32 _b, u32 _s, u32 _h, u32 rms)
	{
		auto rms_diff = [](u32 a, u32 b)
			{
				if (a > b)
					return a - b;
				else
					return b - a;
			};

		u32 x, y;
		for (y = 0; y < lm.height; y++)
		{
			for (x = 0; x < lm.width; x++)
			{
				u32	offset = y * lm.width + x;
				if (lm.marker[offset] >= 254) {
					u8			r, g, b, s, h;
					lm.Pixel(offset, r, g, b, s, h);
					if (rms_diff(_r, r) > rms)				return false;
					if (rms_diff(_g, g) > rms)				return false;
					if (rms_diff(_b, b) > rms)				return false;
					if (rms_diff(_s, s) > rms)				return false;
					if (rms_diff(_h, h) > ((rms * 4) / 3))	return false;
				}
			}
		}
		return true;
	};

	// Average color
	base_color_c	_c;
	u32				_count = rms_average(lm, _c);

	if (0 == _count) {
		clMsg("* ERROR: Lightmap not calculated (T:%d)");
		return	false;
	}
	else		_c.scale(_count);

	// Compress if needed
	u8	_r = u8_clr(_c.rgb.x); //.
	u8	_g = u8_clr(_c.rgb.y);
	u8	_b = u8_clr(_c.rgb.z);
	u8	_s = u8_clr(_c.sun);
	u8	_h = u8_clr(_c.hemi);
	if (rms_test(lm, _r, _g, _b, _s, _h, rms))
	{
		u32 BORDER = gCompilerMode.LC_lmap_BORDER;
		u32		c_x = BORDER * 2;
		u32		c_y = BORDER * 2;
		base_color ccc;		ccc._set(_c);
		lm.surface.assign(c_x * c_y, ccc);
		lm.marker.assign(c_x * c_y, 255);
		lm.height = 0;
		lm.width = 0;
		return true;
	}
	return false;
}