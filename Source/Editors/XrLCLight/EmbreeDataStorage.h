#pragma once
#include "stdafx.h"
#include "R_light.h"


struct RayOptimizedTyped
{
	int type = 0;
	float sqD = 0;
	R_Light* Light;
	base_color_c* Color;
	Face* skip;

	Fvector pos;
	Fvector dir;

	float tmax;
	float tmin;
};

struct RayOptimizedCPU
{
	Fvector pos;
	Fvector dir;

	float tmax;
	float tmin;

	RayOptimizedCPU()
	{

	};

	RayOptimizedCPU(RayOptimizedTyped* ray)
	{
		pos = ray->pos;
		dir = ray->dir;
		tmax = ray->tmax;
		tmin = ray->tmin;
	}
};

struct Hits_XR
{
	float u, v;
	float dist;
	u32 prim;
};
