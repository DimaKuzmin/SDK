#pragma once

enum
{
	LP_DEFAULT			= 0,
	LP_UseFaceDisable	= (1<<0),
	LP_dont_rgb			= (1<<1),
	LP_dont_hemi		= (1<<2),
	LP_dont_sun			= (1<<3),
};

enum LightSource
{
	LS_UNKNOWN,
	LS_RGB,
	LS_SUN,
	LS_HEMI
};
 