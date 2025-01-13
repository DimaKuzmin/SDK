#include "stdafx.h"
#pragma hdrstop

void CBackend::dbg_DP(D3DPRIMITIVETYPE pt, ref_geom geom, u32 vBase, u32 pc)
{
	RCache.set_Geometry		(geom);
	RCache.Render			(pt,vBase,pc);
}

void CBackend::dbg_DIP(D3DPRIMITIVETYPE pt, ref_geom geom, u32 baseV, u32 startV, u32 countV, u32 startI, u32 PC)
{
	RCache.set_Geometry		(geom);
	RCache.Render			(pt,baseV,startV,countV,startI,PC);
}
 