#include "stdafx.h"
//.#include "../xrCore/xrCore.h"
#pragma hdrstop

#include "xrCDB.h"
#include <execution>
#include <atomic>

#include "tbb/tbb.h"

typedef xr_vector<u32>		DWORDList;

bool ArenaCreated = false;
tbb::task_arena* arena;

void CreateArenaTBB()
{
	tbb::task_arena::constraints data;
	data.numa_id = 0;
	// data.core_type;
	data.max_concurrency = 16;
	data.max_threads_per_core = 2;

	arena = new tbb::task_arena (data);  
}
 
u32 parallel_find_if(xr_vector<Fvector>& verts, DWORDList* vl, const Fvector& pred)
{
	if (!ArenaCreated)
	{
		CreateArenaTBB();
		ArenaCreated = true;
	}

	std::atomic<bool> found(false);  // Flag to indicate if the element is found
	std::atomic<u32> result = 0; // Initialize with end iterator

	{
		tbb::blocked_range<u32> range_val(0, vl->size(), 5);

		arena->execute([&] ()
			
			{ 
				tbb::parallel_for(range_val, [&](const tbb::blocked_range<u32>& range)
					{

					
					for (u32 it = range.begin(); it < range.end(); it++)
					{
						if (found.load())
						{
							//Msg_IN_FILE("ID [%d] NOT EXIT", ID);
							tbb::task::current_context()->cancel_group_execution();
							break;
						}


						{
							u32 Index = (*vl)[it];

							if (verts.size() < Index)
   								break;
 
							if (verts[Index].similar(pred))
							{
								result.store(Index);
								found.store(true); // Set the flag to true
								//return; // Exit the loop
							}
						}
					}


				});
					

			}
		);

	}


	u32 r = result.load();
	if (r > verts.size())
	{
		Msg("Vertex Is Dont correct: %u", r);
		return 0;
	}

	return r;
}

namespace CDB
{
	u32		Collector::VPack	(const Fvector& V, float eps)
	{
		xr_vector<Fvector>::iterator I,E;
		I=verts.begin();	E=verts.end();
		
		verts.push_back		(V);
		return verts.size	()-1;
	}

	void	Collector::add_face_D	(
		const Fvector& v0, const Fvector& v1, const Fvector& v2,	// vertices
		u32 dummy								// misc
		)
	{
		TRI T;
		T.verts		[0] = verts.size();
		T.verts		[1] = verts.size()+1;
		T.verts		[2] = verts.size()+2;  
		T.dummy			= dummy;

		verts.push_back(v0);
		verts.push_back(v1);
		verts.push_back(v2);
		faces.push_back(T);
	}

	void	Collector::add_face		(	const Fvector& v0, const Fvector& v1, const Fvector& v2, u16 material, u16 sector )
	{
		TRI			T;
		T.verts	[0]		= verts.size();
		T.verts	[1]		= verts.size()+1;
		T.verts	[2]		= verts.size()+2;
		T.material		= material;
		T.sector		= sector;

		verts.push_back(v0);
		verts.push_back(v1);
		verts.push_back(v2);
		faces.push_back(T);
	}

	void	Collector::add_face_packed	(
		const Fvector& v0, const Fvector& v1, const Fvector& v2,	// vertices
		u16		material, u16 sector,								// misc
		float	eps
		)
	{
		TRI T;
		T.verts	[0]		= VPack(v0,eps);
		T.verts	[1]		= VPack(v1,eps);
		T.verts	[2]		= VPack(v2,eps);
		T.material		= material;
		T.sector		= sector;
		faces.push_back(T);
	}

	void	Collector::add_face_packed_D	(
		const Fvector& v0, const Fvector& v1, const Fvector& v2,	// vertices
		u32		dummy,	float eps
		)
	{
		TRI T;
		T.verts	[0] = VPack(v0,eps);
		T.verts	[1] = VPack(v1,eps);
		T.verts	[2] = VPack(v2,eps);
		T.dummy			= dummy;
		faces.push_back(T);
	}

#pragma warning(push)
#pragma warning(disable:4995)
#include <malloc.h>
#pragma warning(pop)

#pragma pack(push,1)
	struct edge {
		u32		face_id		: 30;
		u32		edge_id		: 2;
		u16		vertex_id0;
		u16		vertex_id1;
	};
#pragma pack(pop)

	struct sort_predicate {
		IC	bool	operator()	(const edge &edge0, const edge &edge1) const
		{
			if (edge0.vertex_id0 < edge1.vertex_id0)
				return				(true);

			if (edge1.vertex_id0 < edge0.vertex_id0)
				return				(false);

			if (edge0.vertex_id1 < edge1.vertex_id1)
				return				(true);

			if (edge1.vertex_id1 < edge0.vertex_id1)
				return				(false);

			return					(edge0.face_id < edge1.face_id);
		}
	};

	void	Collector::calc_adjacency	(xr_vector<u32>& dest)
	{
 
		VERIFY							(faces.size() < 65536);
		const u32						edge_count = faces.size()*3;
#ifdef _EDITOR
		xr_vector<edge> _edges			(edge_count);
		edge 							*edges = &*_edges.begin();
#else
		edge							*edges = (edge*)_alloca(edge_count*sizeof(edge));
#endif
		edge							*i = edges;
		xr_vector<TRI>::const_iterator	B = faces.begin(), I = B;
		xr_vector<TRI>::const_iterator	E = faces.end();
		for ( ; I != E; ++I) {
			u32							face_id = u32(I - B);

			(*i).face_id				= face_id;
			(*i).edge_id				= 0;
			(*i).vertex_id0				= (u16)(*I).verts[0];
			(*i).vertex_id1				= (u16)(*I).verts[1];
			if ((*i).vertex_id0 > (*i).vertex_id1)
				std::swap				((*i).vertex_id0,(*i).vertex_id1);
			++i;
			
			(*i).face_id				= face_id;
			(*i).edge_id				= 1;
			(*i).vertex_id0				= (u16)(*I).verts[1];
			(*i).vertex_id1				= (u16)(*I).verts[2];
			if ((*i).vertex_id0 > (*i).vertex_id1)
				std::swap				((*i).vertex_id0,(*i).vertex_id1);
			++i;
			
			(*i).face_id				= face_id;
			(*i).edge_id				= 2;
			(*i).vertex_id0				= (u16)(*I).verts[2];
			(*i).vertex_id1				= (u16)(*I).verts[0];
			if ((*i).vertex_id0 > (*i).vertex_id1)
				std::swap				((*i).vertex_id0,(*i).vertex_id1);
			++i;
		}

		std::sort						(edges,edges + edge_count,sort_predicate());

		dest.assign						(edge_count,u32(-1));

		{
			edge						*I = edges, *J;
			edge						*E = edges + edge_count;
			for ( ; I != E; ++I) {
				if (I + 1 == E)
					continue;

				J							= I + 1;

				if ((*I).vertex_id0 != (*J).vertex_id0)
					continue;

				if ((*I).vertex_id1 != (*J).vertex_id1)
					continue;

				dest[(*I).face_id*3 + (*I).edge_id]	= (*J).face_id;
				dest[(*J).face_id*3 + (*J).edge_id]	= (*I).face_id;
			}
		}
	}

    IC BOOL similar(TRI& T1, TRI& T2)
    {
        if ((T1.verts[0]==T2.verts[0]) && (T1.verts[1]==T2.verts[1]) && (T1.verts[2]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        if ((T1.verts[0]==T2.verts[0]) && (T1.verts[2]==T2.verts[1]) && (T1.verts[1]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        if ((T1.verts[2]==T2.verts[0]) && (T1.verts[0]==T2.verts[1]) && (T1.verts[1]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        if ((T1.verts[2]==T2.verts[0]) && (T1.verts[1]==T2.verts[1]) && (T1.verts[0]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        if ((T1.verts[1]==T2.verts[0]) && (T1.verts[0]==T2.verts[1]) && (T1.verts[2]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        if ((T1.verts[1]==T2.verts[0]) && (T1.verts[2]==T2.verts[1]) && (T1.verts[0]==T2.verts[2]) && (T1.dummy==T2.dummy)) return TRUE;
        return FALSE;
    }

    void Collector::remove_duplicate_T( )
    {
		for (u32 f=0; f<faces.size(); f++)
		{
			for (u32 t=f+1; t<faces.size();)
			{
				if (t==f)	continue;
                TRI& T1	= faces[f];
                TRI& T2	= faces[t];
                if (similar(T1,T2)){
                	faces[t] = faces.back();
                    faces.pop_back();
                }else{
                	t++;
                }
            }
        }
    }


	CollectorPacked::CollectorPacked(const Fbox &bb, int apx_vertices, int apx_faces)
	{
		// Params
		VMscale.set		(bb.max.x-bb.min.x, bb.max.y-bb.min.y, bb.max.z-bb.min.z);
		VMmin.set		(bb.min);
		VMeps.set		(VMscale.x/clpMX/2,VMscale.y/clpMY/2,VMscale.z/clpMZ/2);
		VMeps.x			= (VMeps.x<EPS_L)?VMeps.x:EPS_L;
		VMeps.y			= (VMeps.y<EPS_L)?VMeps.y:EPS_L;
		VMeps.z			= (VMeps.z<EPS_L)?VMeps.z:EPS_L;

		// Preallocate memory
		verts.reserve	(apx_vertices);
		faces.reserve	(apx_faces);
		flags.reserve	(apx_faces);
		int		_size	= (clpMX+1)*(clpMY+1)*(clpMZ+1);
		int		_average= (apx_vertices/_size)/2;

		for (int TH=0; TH < MAX_THREADS; TH++)
		for (int ix=0; ix<clpMX+1; ix++)
		for (int iy=0; iy<clpMY+1; iy++)
		for (int iz=0; iz<clpMZ+1; iz++)
			VM[TH][ix][iy][iz].reserve	(_average);
	}

	void	CollectorPacked::add_face(
		const Fvector& v0, const Fvector& v1, const Fvector& v2,	// vertices
		u16 material, u16 sector, u32 _flags, u32 TH									// misc
		)
	{
		TRI T;
		T.verts	[0] = VPack(v0, TH);
		T.verts	[1] = VPack(v1, TH);
		T.verts	[2] = VPack(v2, TH);
		T.material		= material;
		T.sector		= sector;
		flags.push_back(_flags);
		faces.push_back(T);

	}

	void	CollectorPacked::add_face_D(
		const Fvector& v0, const Fvector& v1, const Fvector& v2,	// vertices
		u32 dummy, u32 _flags, u32 TH										// misc
		)
	{
		TRI T;
		T.verts	[0] = VPack(v0, TH);
		T.verts	[1] = VPack(v1, TH);
		T.verts	[2] = VPack(v2, TH);
		T.dummy		= dummy;
		faces.push_back(T);
		flags.push_back(_flags);
	}

	xrCriticalSection xrCDB;

	u32		CollectorPacked::VPack(const Fvector& V, int TH)
	{
		u32 P = 0xffffffff;

		u32 ix,iy,iz;
		ix = iFloor(float(V.x-VMmin.x)/VMscale.x*clpMX);
		iy = iFloor(float(V.y-VMmin.y)/VMscale.y*clpMY);
		iz = iFloor(float(V.z-VMmin.z)/VMscale.z*clpMZ);

		//		R_ASSERT(ix<=clpMX && iy<=clpMY && iz<=clpMZ);
		clamp(ix,(u32)0,clpMX);	clamp(iy,(u32)0,clpMY);	clamp(iz,(u32)0,clpMZ);
		 
 		if (UsePacking)
		{
			xrCDB.Enter();
			DWORDList* vl = &(VM[TH][ix][iy][iz]);;
			xrCDB.Leave();			
			
			if (true)
			for (DWORDIt it = vl->begin(); it != vl->end(); it++)
			{
				if (*it >= verts.size())
					break;

				if (verts[*it].similar(V))
				{
					P = *it;
					break;
				}
			}
			 
		}

		if (0xffffffff==P)
		{
			P = verts.size();
			xrCDB.Enter();
			verts.push_back(V);

			VM[TH][ix][iy][iz].push_back(P);

			u32 ixE,iyE,izE;
			ixE = iFloor(float(V.x+VMeps.x-VMmin.x)/VMscale.x*clpMX);
			iyE = iFloor(float(V.y+VMeps.y-VMmin.y)/VMscale.y*clpMY);
			izE = iFloor(float(V.z+VMeps.z-VMmin.z)/VMscale.z*clpMZ);

			//			R_ASSERT(ixE<=clpMX && iyE<=clpMY && izE<=clpMZ);
			clamp(ixE,(u32)0,clpMX);	
			clamp(iyE,(u32)0,clpMY);	
			clamp(izE,(u32)0,clpMZ);

			if (ixE!=ix)							VM[TH][ixE][iy][iz].push_back	(P);
			if (iyE!=iy)							VM[TH][ix][iyE][iz].push_back	(P);
			if (izE!=iz)							VM[TH][ix][iy][izE].push_back	(P);
			if ((ixE!=ix)&&(iyE!=iy))				VM[TH][ixE][iyE][iz].push_back	(P);
			if ((ixE!=ix)&&(izE!=iz))				VM[TH][ixE][iy][izE].push_back	(P);
			if ((iyE!=iy)&&(izE!=iz))				VM[TH][ix][iyE][izE].push_back	(P);
			if ((ixE!=ix)&&(iyE!=iy)&&(izE!=iz))	VM[TH][ixE][iyE][izE].push_back	(P);

			xrCDB.Leave();
		}
		return P;
	}

	void	CollectorPacked::clear()
	{
		verts.clear_and_free	();
		faces.clear_and_free	();
		flags.clear_and_free	();
		
		for (u32 TH = 0; TH < MAX_THREADS; TH++)
		for (u32 _x=0; _x<=clpMX; _x++)
		for (u32 _y=0; _y<=clpMY; _y++)
		for (u32 _z=0; _z<=clpMZ; _z++)
			VM[TH][_x][_y][_z].clear_and_free	();
	}
};
