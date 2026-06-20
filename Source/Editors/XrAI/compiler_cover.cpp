#include "stdafx.h"
#include "compiler.h"
#include "compiler_covers_helpers.h"

#include "cl_intersect.h"
#include "quadtree.h"
#include "EmbreeRayTracing.h"
#include <ppl.h>
 
Shader_xrLC_LIB*				g_shaders_xrlc	;
xr_vector<b_material>			g_materials		;
xr_vector<b_shader>				g_shader_render	;
xr_vector<b_shader>				g_shader_compile;
xr_vector<b_BuildTexture>		*g_textures	=nullptr	;
xr_vector<FaceDataEmbree>		g_embree_faces;	

xr_vector<bool>					g_cover_nodes;
 
// Compute Cover 
typedef float	Cover[4];

void ComputeCover(xrCoverHelper::Query& Q, u32 N)
{
	auto compute_cover_value = [](xrCoverHelper::Query& Q, u32 const& N, vertex& BaseNode, float const& cover_height, Cover& cover)
		{
			Fvector& BasePos = BaseNode.Pos;
			Fvector	 TestPos = BasePos; TestPos.y += cover_height;

			float	c_total[8] = { 0,0,0,0,0,0,0,0 };
			float	c_passed[8] = { 0,0,0,0,0,0,0,0 };

			// perform volumetric query
			Q.Init(BasePos);
			Q.Perform(N);

			// main cycle: trace rays and compute counts
			for (auto it = Q.q_List.begin(); it != Q.q_List.end(); it++)
			{
				// calc dir & range
				u32		ID = *it;
				R_ASSERT(ID < g_nodes.size());
				if (N == ID)		continue;
				vertex& N = g_nodes[ID];
				Fvector& Pos = N.Pos;
				Fvector		Dir;
				Dir.sub(Pos, BasePos);
				float		range = Dir.magnitude();
				Dir.div(range);

				// raytrace
				int			sector = xrCoverHelper::calcSphereSector(Dir);
				c_total[sector] += 1.f;

				extern SceneEmbreeAI			 SceneEmbreeInterface;
				c_passed[sector] += SceneEmbreeInterface.RayTrace(TestPos, Dir, range);
			}
			Q.Clear();

			// analyze probabilities
			float	value[8];
			for (int dirs = 0; dirs < 8; dirs++) {
				R_ASSERT(c_passed[dirs] <= c_total[dirs]);
				if (c_total[dirs] == 0)	value[dirs] = 0;
				else					value[dirs] = float(c_passed[dirs]) / float(c_total[dirs]);
				clamp(value[dirs], 0.f, 1.f);
			}

			if (value[0] < .999f) {
				value[0] = value[0];
			}

			cover[0] = (value[2] + value[3] + value[4] + value[5]) / 4.f; clamp(cover[0], 0.f, 1.f);	// left
			cover[1] = (value[0] + value[1] + value[2] + value[3]) / 4.f; clamp(cover[1], 0.f, 1.f);	// forward
			cover[2] = (value[6] + value[7] + value[0] + value[1]) / 4.f; clamp(cover[2], 0.f, 1.f);	// right
			cover[3] = (value[4] + value[5] + value[6] + value[7]) / 4.f; clamp(cover[3], 0.f, 1.f);	// back
		};
	

	// initialize process
	vertex& BaseNode = g_nodes[N];
	if (!g_cover_nodes[N])
	{
		BaseNode.high_cover[0] = flt_max;
		BaseNode.high_cover[1] = flt_max;
		BaseNode.high_cover[2] = flt_max;
		BaseNode.high_cover[3] = flt_max;
		BaseNode.low_cover[0] = flt_max;
		BaseNode.low_cover[1] = flt_max;
		BaseNode.low_cover[2] = flt_max;
		BaseNode.low_cover[3] = flt_max;
		return;
	}

	compute_cover_value(Q, N, BaseNode, high_cover_height, BaseNode.high_cover);
	compute_cover_value(Q, N, BaseNode, low_cover_height, BaseNode.low_cover);
}


void	xrCover	(bool pure_covers)
{
	Status("Calculating...");

	if (!pure_covers)
		compute_cover_nodes	();
	else
		g_cover_nodes.assign(g_nodes.size(),true);

	// Start threads, wait, continue --- perform all the work

	static std::atomic<u32> CurrentPos; CurrentPos = 0;

	concurrency::parallel_for(size_t(0), size_t(gCompilerMode.ThreadsNum), [](size_t ThreadID) 
	{
		thread_local xrCoverHelper::Query Q;
 		Q.Begin(g_nodes.size());

		for (;;)
		{
			int N = CurrentPos.fetch_add(1);
			if (N >= g_nodes.size()) break;

			AditionalData("Node : %u/%u", N, g_nodes.size());
			ComputeCover(Q, N);

		}

	});

 
	Status("Calculating non covers...");
	if (!pure_covers) 
	{
		compute_non_covers	();
		return;
	}

	// Smooth
	Status			("Smoothing coverage mask...");
 
	Nodes	Old		= g_nodes;
	for (u32 N=0; N<g_nodes.size(); N++)
	{
		vertex&	Base		= Old[N];
		vertex&	Dest		= g_nodes[N];
		
		for (int dir=0; dir<4; dir++)
		{
			float val		= 2*Base.high_cover[dir];
			float val2		= 2*Base.low_cover[dir];
			float cnt		= 2;
			
			for (int nid=0; nid<4; nid++) {
				if (Base.n[nid]!=InvalidNode) {
					val		+=  Old[Base.n[nid]].high_cover[dir];
					val2	+=  Old[Base.n[nid]].low_cover[dir];
					cnt		+=	1.f;
				}
			}
			Dest.high_cover[dir]	=  val/cnt;
			Dest.low_cover[dir]		=  val2/cnt;
		}
	}
}
