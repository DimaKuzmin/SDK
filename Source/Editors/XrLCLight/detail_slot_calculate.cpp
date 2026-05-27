////////////////////////////////////////////////////////////////////////////
//	Created		: 27.03.2009
//	Author		: Konstantin Slipchenko
//	Copyright (C) GSC Game World - 2009
////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "detail_slot_calculate.h"

#include "cl_intersect.h"
#include "base_lighting.h"
#include "global_calculation_data.h"
#include "../Public/shader_xrlc.h"
#include "embree_raytracing/EmbreeRayTrace.h"
#include "light_point.h"
#include "xrDeflector.h"

//-----------------------------------------------------------------------------------------------------------------
XRLC_LIGHT_API extern int	LIGHT_Count				=	7;
    
bool detail_slot_calculate( u32 _x, u32 _z, DetailSlot&	DS)
{
	process_pallete(DS);
	if (gl_data.slots_data.skip_slot(_x, _z)) return false;
 
	thread_local xr_vector<u32> box_result;
	thread_local CDB::COLLIDER DB;
	thread_local base_lighting Selected;

	///////////////////////////////////////////////////////////
	// Build slot BB & sphere
	Fbox	BB;
	gl_data.slots_data.get_slot_box( BB, _x, _z );

	Fsphere		S;
	BB.getsphere( S.P, S.R );

	// Select polygons
	Fvector				bbC,bbD;
	BB.get_CD			( bbC, bbD );	bbD.add( 0.01f );
	DB.box_query		( &gl_data.RCAST_Model, bbC, bbD );
	

	box_result.clear	();
	for (CDB::RESULT* I=DB.r_begin(); I!=DB.r_end(); I++) 
		box_result.push_back(I->id);
	
	if (box_result.empty())	
		return false; 

	CDB::TRI*	tris	= gl_data.RCAST_Model.get_tris();
	Fvector*	verts	= gl_data.RCAST_Model.get_verts();

	// select lights
	Selected.select		( gl_data.g_lights, S.P, S.R );

	// lighting itself
	base_color_c		amount;
	u32					count	= 0;
	float coeff		= DETAIL_SLOT_SIZE_2/float(LIGHT_Count);
 	
 	for (int x=-LIGHT_Count; x<=LIGHT_Count; x++) 
	{
		Fvector		P;
		P.x			= bbC.x + coeff*float(x);

		for (int z=-LIGHT_Count; z<=LIGHT_Count; z++) 
		{
			// compute position
			Fvector t_n;	t_n.set(0,1,0);
			P.z				= bbC.z + coeff*float(z);
			P.y				= BB.min.y-5;
			Fvector	dir;	dir.set		(0,-1,0);
			Fvector start;	start.set	(P.x,BB.max.y+EPS,P.z);
			
			float		r_u,r_v,r_range;
			
			
			for (auto tit = box_result.begin(); tit!=box_result.end(); tit++)
			{
			
				CDB::TRI&	T		= tris	[*tit];
				Fvector		V[3]	= { verts[T.verts[0]], verts[T.verts[1]], verts[T.verts[2]] };
				
				// Fast Check (se7kills)
				if (CDB::TestRayTri(start,dir,V,r_u,r_v,r_range,TRUE))
				{
					if (r_range>=0.f)	
					{
						float y_test	= start.y - r_range;
						if (y_test>P.y)
						{
							P.y			= y_test+EPS;
							t_n.mknormal(V[0],V[1],V[2]);
						}
					}
				}
			}	
			

			if (P.y<BB.min.y) continue;
			
			// light point
			LightPoint		 ( amount, P, t_n, Selected, 0, 0);
			count			+= 1;
		}
	}

	// calculation of luminocity
	amount.scale		(count);
	amount.mul			(.5f);
	DS.c_dir			= DS.w_qclr	(amount.sun,15);
	DS.c_hemi			= DS.w_qclr	(amount.hemi,15);
	DS.c_r				= DS.w_qclr	(amount.rgb.x,15);
	DS.c_g				= DS.w_qclr	(amount.rgb.y,15);
	DS.c_b				= DS.w_qclr	(amount.rgb.z,15);
	////////////////////////////////////////////////////////////
	return true;
}

#include <ppl.h>
extern bool useDetails;

void xrCompileDO(u32 Samples)
{
	Phase("Loading level...");
	gl_data.xrLoad();

	Phase("Lighting nodes...");
	CDB::COLLIDER		DB;
	DB.ray_options(CDB::OPT_CULL);
	DB.box_options(CDB::OPT_FULL_TEST);
	base_lighting		Selected;

	static std::atomic<u32> atomic_task;
	CTimer start_time; start_time.Start();

	// Lightpoint поментка чтобы использовал алогоритм с Details !
	useDetails = true;
	concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [](size_t tID)
		{
			while (true)
			{
				u32 Z = atomic_task.fetch_add(1);
 				if (Z >= gl_data.slots_data.size_z()) break;

				AditionalData("Process: %u/%u", Z, gl_data.slots_data.size_z());

				for (u32 X = 0; X < gl_data.slots_data.size_x(); X++)
				{
					DetailSlot& DS = gl_data.slots_data.get_slot(X, Z);
 					detail_slot_calculate(X, Z, DS);
 				}
			}
 		}
	);

	useDetails = false;

	Msg("%d seconds elapsed.", (start_time.GetElapsed_ms()) / 1000);

	gl_data.slots_data.Free();
}