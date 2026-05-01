#pragma once

enum
{
	LP_DEFAULT			= 0,
	LP_UseFaceDisable	= (1<<0),
	LP_dont_rgb			= (1<<1),
	LP_dont_hemi		= (1<<2),
	LP_dont_sun			= (1<<3),
};

static u32 GetCurrentFlags()
{
	return  (gCompilerMode.LC_NoRGB ? LP_dont_rgb : 0) | 
			(gCompilerMode.LC_NoSun ? LP_dont_sun : 0) |
			(gCompilerMode.LC_NoHemi ? LP_dont_hemi : 0);
}

