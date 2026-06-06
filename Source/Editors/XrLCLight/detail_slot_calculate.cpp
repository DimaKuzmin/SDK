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
#include "light_point.h"
#include "xrDeflector.h"

// Cuda - Embree
#include "embree_raytracing/EmbreeRayTrace.h"
#include "CUDA/xrDeflectorLight_Packed.h"

//-----------------------------------------------------------------------------------------------------------------
XRLC_LIGHT_API extern int	LIGHT_Count				=	7;
    
bool detail_slot_calculate( u32 _x, u32 _z)
{
	DetailSlot& DS = gl_data.slots_data.get_slot(_x, _z);
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

	R_ASSERT(gl_data.RCAST_Model);

	// Select polygons
	Fvector				bbC,bbD;
	BB.get_CD			( bbC, bbD );	bbD.add( 0.01f );
	DB.box_query		( gl_data.RCAST_Model, bbC, bbD );
	

	box_result.clear	();
	for (CDB::RESULT* I=DB.r_begin(); I!=DB.r_end(); I++) 
		box_result.push_back(I->id);
	
	if (box_result.empty())	
		return false; 

	CDB::TRI*	tris	= gl_data.RCAST_Model->get_tris();
	Fvector*	verts	= gl_data.RCAST_Model->get_verts();

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
			if (gCompilerMode.Embree)
				LightPoint(amount, P, t_n, Selected, 0, 0);
 			else if (gCompilerMode.CUDA)
			{
				size_t idx = GPUTaskinSystem.MakeKey(_x, _z);
				GPUTaskinSystem.LightPointPacked_add_task(idx, nullptr, P, t_n, nullptr);
			}
  
			count			+= 1;
		}
	}

	if (gCompilerMode.Embree)
	{
		// calculation of luminocity
		amount.scale(count);
		amount.mul(.5f);
		DS.c_dir  = DS.w_qclr(amount.sun, 15);
		DS.c_hemi = DS.w_qclr(amount.hemi, 15);
		DS.c_r    = DS.w_qclr(amount.rgb.x, 15);
		DS.c_g    = DS.w_qclr(amount.rgb.y, 15);
		DS.c_b    = DS.w_qclr(amount.rgb.z, 15);
	}

	////////////////////////////////////////////////////////////
	return true;
}

#include <ppl.h>
extern bool useDetails;

xr_vector<u32>			 samples;
xr_vector<base_color_c>  detail_colors;
u32 size_x;
u32 size_z;

void ApplyColorDetailGPU(size_t IndexTask, base_color_c& C)
{
	u32 x = GPUTaskinSystem.GetU(IndexTask);
	u32 z = GPUTaskinSystem.GetV(IndexTask);

	u32 idx = z * size_x + x;
	samples[idx]++;
	detail_colors[idx].add(C);
}

void ApplyColorsGPU()
{
	for (auto x = 0; x < gl_data.slots_data.size_x(); x++)
	for (auto z = 0; z < gl_data.slots_data.size_z(); z++)
	{
		// Getter - Detail Slot
		auto& DS = gl_data.slots_data.get_slot(x, z);

		u32 idx = z * size_x + x;
		auto& count = samples[idx];
		if (count > 0)
		{
			auto& color = detail_colors[idx];
 			color.scale(count);
			color.mul(.5f);
			
			// Пишется результат в (level.details) !
			DS.c_dir  = DS.w_qclr(color.sun, 15);
			DS.c_hemi = DS.w_qclr(color.hemi, 15);
			DS.c_r    = DS.w_qclr(color.rgb.x, 15);
			DS.c_g    = DS.w_qclr(color.rgb.y, 15);
			DS.c_b    = DS.w_qclr(color.rgb.z, 15);
		}
	}

	samples.clear();
	samples.shrink_to_fit();

	detail_colors.clear();
	detail_colors.shrink_to_fit();
}



void SaveAsOBJ(TriangleContainer& Container)
{
	IWriter* W = FS.w_open("$level$", "rcast_model.obj");
 
 	string256 tmp;
	// vertices
	for (auto& V: Container.vertex()) {
 		xr_sprintf(tmp, "v %f %f %f", V.x, V.y, -V.z);
		W->w_string(tmp);
	}
	// transfer faces
	for (auto& TRI : Container.faces())
	{
 		xr_sprintf(tmp, "f %d %d %d", TRI.point1 + 1, TRI.point2 + 1, TRI.point3 + 1);
		W->w_string(tmp);
	}
	FS.w_close(W);
}

void BuildModel(TriangleContainer& container)
{
	for (auto& F : gl_data.building_embree_faces)
	{
		container.AddFaceRaw(&F, F.v1, F.v2, F.v3);
	}
	container.RemoveDublicates(true);
	
	if (gCompilerMode.SaveObjectRcast)
		SaveAsOBJ(container);

	gl_data.RCAST_Model = xr_new<CDB::MODEL>();

	auto& Vert = container.vertex();
 	xr_vector<CDB::TRI> faces;
	for (auto T : container.faces())
		faces.push_back(T.Get());
	gl_data.RCAST_Model->build(Vert.data(), Vert.size(), faces.data(), faces.size(), nullptr);

	faces.clear();
	faces.shrink_to_fit();
}


void xrCompileDO()
{
	Phase("Loading level...");
	gl_data.xrLoad();

	Phase("Building Model...");
	TriangleContainer container;
	BuildModel(container);
 
	if (gCompilerMode.Embree)
	{
		EmbreeMain.InitEmbreeDetails(container);
		container.ClearAll();

		static std::atomic<u32> atomic_task; 
		atomic_task = 0;
		// Lightpoint поментка чтобы использовал алогоритм с Details !
		useDetails = true;


		Phase("Lighting Details...");
		auto Task = []()
			{
				while (true)
				{
					u32 Z = atomic_task.fetch_add(1);
					if (Z >= gl_data.slots_data.size_z()) break;

					AditionalData("Embree Process: %u/%u", Z, gl_data.slots_data.size_z());

					for (u32 X = 0; X < gl_data.slots_data.size_x(); X++)
					{
						detail_slot_calculate(X, Z);
					}
				}
			};
		runThreadsMax(Task, gCompilerMode.ThreadsNum);


		useDetails = false;
	}
	else if (gCompilerMode.CUDA)
	{
		size_x = gl_data.slots_data.size_x();
		size_z = gl_data.slots_data.size_z();

		samples.resize(size_x * size_z);
		detail_colors.resize(size_x * size_z);

		GPUTaskinSystem.InitializeGPU();
 		GPUTaskinSystem.ColorsMapType = eDetails;

		static std::atomic<u32> atomic_task;
		atomic_task = 0;

		auto Task = []()
			{
				while (true)
				{
					u32 Z = atomic_task.fetch_add(1);
					AditionalData("Cuda Process: %u/%u", Z, gl_data.slots_data.size_z());

					if (Z >= gl_data.slots_data.size_z()) break;

					for (u32 X = 0; X < gl_data.slots_data.size_x(); X++)
					{
						detail_slot_calculate(X, Z);
 					}
				}

				GPUTaskinSystem.LightPointPacked_run_tasks();
			};

		runThreadsMax(Task, gCompilerMode.ThreadsNum);
		 
		GPUTaskinSystem.RestartALL();


 		ApplyColorsGPU();
	}

	Phase("Unloading data buffers...");
	gl_data.xrUnload();
	container.ClearAll();

	EmbreeMain.IntelEmbereUnloadData();
	GPUTaskinSystem.CleanupGPU();
}