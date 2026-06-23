#pragma once

#include "xrFace.h"
#include "base_color.h"
#include "lm_layer.h"
#include "uv_tri.h"
#include "../../xrcdb/xrCDB.h"
 
#include "R_light.h"
#include "embree_raytracing/EmbreeRayTrace.h"

class  base_lighting;
class CDeflector;

 
class XRLC_LIGHT_API CDeflector 
{

public:
 	xr_vector<UVtri>			UVpolys;
	Fvector						normal;
	lm_layer					layer;
	Fsphere						Sphere;
	
	bool						bMerged;
	bool						bLightProcessed;
public:

						CDeflector					();
 						~CDeflector					();
 

	void	OA_SetNormal		(Fvector &_N )	{ normal.set(_N); normal.normalize(); VERIFY(_valid(normal)); }
	BOOL	OA_Place			(Face *owner);
	void	OA_Place			(xr_vector<Face*>& lst);
	void	OA_Export			();
		
	void	GetRect				(Fvector2 &min, Fvector2 &max);
	u32		GetFaceCount()		{ return (u32)UVpolys.size();	};
		
	void	Light				( CDB::COLLIDER* DB, base_lighting* LightsSelected );
 								 
	void	L_Direct_Edge		(CDB::COLLIDER* DB, base_lighting* LightsSelected, Fvector2& p1, Fvector2& p2, Fvector& v1, Fvector& v2, Fvector& N, float texel_size, Face* skip);
	void	L_Direct			( CDB::COLLIDER* DB, base_lighting* LightsSelected);

	u32		weight				() { return layer.Area(); }	


	u64		size_deflector() 
	{
		u32 STri	 = UVpolys.capacity() * sizeof(UVtri);
		u32 SLMLayer = layer.memory_lmap();
		 
 		return sizeof(*this) + STri + SLMLayer;
	}
	u16		GetBaseMaterial		() ;

	void	Bounds				(u32 ID, Fbox2& dest)
	{
		UVtri& TC		= UVpolys[ID];
		dest.min.set	(TC.uv[0]);
		dest.max.set	(TC.uv[0]);
		dest.modify		(TC.uv[1]);
		dest.modify		(TC.uv[2]);
	}

	void	Bounds_Summary		(Fbox2& bounds)
	{
		bounds.invalidate();
		for (u32 I=0; I<UVpolys.size(); I++)
		{
			Fbox2	B;
			Bounds	(I, B);
			bounds.merge(B);
		}

		if (bounds.min.x == bounds.max.x || bounds.min.y == bounds.max.y)
		{
			Msg("! Deflector bounds min[%f][%f] max[%f][%f]",
				bounds.min.x, bounds.max.y,
				bounds.max.x, bounds.max.y
			);

			for (auto F : UVpolys)
			{
				Fvector C;
				F.owner->CalcCenter(C);
				Msg("! Face Pos error {%f, %f, %f}", VPUSH(C) );
			}

		}

		R_ASSERT(bounds.min.x != bounds.max.x);
		R_ASSERT(bounds.min.y != bounds.max.y);
	}

	void	RemapUV				(xr_vector<UVtri>& dest, u32 base_u, u32 base_v, u32 size_u, u32 size_v, u32 lm_u, u32 lm_v, BOOL bRotate);
	void	RemapUV				(u32 base_u, u32 base_v, u32 size_u, u32 size_v, u32 lm_u, u32 lm_v, BOOL bRotate);
	  	
	bool	similar				( const CDeflector &D, float eps =EPS ) const;
	bool	similar_pos				( const CDeflector &D, float eps =EPS ) const;

	// GPU CODE:
	// Stage 1
	void LightGPU();
	void L_DirectGPU();

	// cuda recvest color reciver
	u32 ProcessedUVColors;
	bool ApplyColors();
	void ApplyColor(size_t INDEX, base_color_c& C);

	// Stage 2
	void ApplyExpandBordersGPU();


	// Clearing Memory
	void DealocateMemory() { layer.clear_memory(); };
};

extern XRLC_LIGHT_API void		Jitter_Select	(Fvector2* &Jitter, u32& Jcount);
extern void		blit			(u32* dest,		u32 ds_x, u32 ds_y, u32* src,		u32 ss_x, u32 ss_y, u32 px, u32 py, u32 aREF);
extern XRLC_LIGHT_API void		blit			(lm_layer& dst, u32 ds_x, u32 ds_y, lm_layer& src,	u32 ss_x, u32 ss_y, u32 px, u32 py, u32 aREF);
extern void		blit_r			(u32* dest,		u32 ds_x, u32 ds_y, u32* src,		u32 ss_x, u32 ss_y, u32 px, u32 py, u32 aREF);
extern XRLC_LIGHT_API void		blit_r			(lm_layer& dst, u32 ds_x, u32 ds_y, lm_layer& src,	u32 ss_x, u32 ss_y, u32 px, u32 py, u32 aREF);
extern void		lblit			(lm_layer& dst, lm_layer& src, u32 px, u32 py, u32 aREF);

extern XRLC_LIGHT_API void		LightPoint(base_color_c& C, Fvector& P, Fvector& N, base_lighting& lights, u32 flags, Face* skip);
extern XRLC_LIGHT_API void		LightPoint_Embree(EmbreeRayTraceModel* MDL, base_color_c& C, Fvector& P, Fvector& N, base_lighting& lights, u32 flags, Face* skip);
 
#define rms_zero	((4+g_params().m_lm_rms_zero)/2)
#define rms_shrink	((8+g_params().m_lm_rms)/2)
 