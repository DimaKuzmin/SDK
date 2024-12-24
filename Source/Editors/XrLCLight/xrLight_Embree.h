#pragma once

#include "R_light.h"
#include "base_lighting.h"
#include "base_color.h"
 
extern XRLC_LIGHT_API float RaytraceEmbreeProcess(R_Light& L, Fvector& P, Fvector& N, float range, void* skip);
extern XRLC_LIGHT_API void LightPointEmbree(base_color_c& C, Fvector& P, Fvector& N, base_lighting& lights, u32 flags, void* skip);