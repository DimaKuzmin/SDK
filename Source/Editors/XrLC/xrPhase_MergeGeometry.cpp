#include "stdafx.h"

#include "build.h"
#include "../xrLCLight/xrface.h"

#include <thread>

extern void Detach		(vecFace* S);
  
IC BOOL	FaceEqual		(Face* F1, Face* F2)
{
	if (F1->v[0]->P.distance_to(F2->v[0]->P) > 64 ) return FALSE;
	if (F1->dwMaterial  != F2->dwMaterial)		return FALSE;
	if (F1->tc.size()	!= F2->tc.size())		return FALSE;
	if (F1->lmap_layer  != F2->lmap_layer)		return FALSE;
	return TRUE;
}

#include <mutex>
std::mutex lock;

xr_map<u32, Fbox > bb_bases;
  
ICF void CreateBox(vecFace& subdiv, Fbox& bb_base, u32 id)
{
	if (bb_bases.find(id) == bb_bases.end())
	{
		for (u32 it = 0; it < subdiv.size(); it++)
		{
			Face* F = subdiv[it];
			bb_base.modify(F->v[0]->P);
			bb_base.modify(F->v[1]->P);
			bb_base.modify(F->v[2]->P);
			 
		}
		lock.lock();
		bb_bases[id] = bb_base;
		lock.unlock();
	}
	else
	{
		bb_base = bb_bases[id];
	}
}
 
BOOL	NeedMerge		(vecFace& subdiv, Fbox& bb_base, u32 id)
{


	// 1. Amount of polygons
	if (subdiv.size()>=u32(3*c_SS_HighVertLimit/4))	return FALSE;
	  
	Fvector sz_base;

 	// 2. Bounding box
	bb_base.invalidate	();
	CreateBox(subdiv, bb_base, id);
	
	bb_base.grow		(EPS_S);	// Enshure non-zero volume
	bb_base.getsize(sz_base);
 

	
	if (sz_base.x<c_SS_maxsize)		return TRUE;
	if (sz_base.y<c_SS_maxsize)		return TRUE;
	if (sz_base.z<c_SS_maxsize)		return TRUE;
	return FALSE;
}
 
IC void	MakeCube		(Fbox& BB_dest, const Fbox& BB_src)
{
	Fvector C,D;
	BB_src.get_CD		(C,D);
	float	max			= D.x;
	if (D.y>max)	max = D.y;
	if (D.z>max)	max = D.z;

	BB_dest.set			(C,C);
	BB_dest.grow		(max); 
}

IC BOOL ValidateMergeLinearSize( const Fvector & merged, const Fvector & orig1, const Fvector & orig2, int iAxis)
{
	if ( ( merged[iAxis] > (4*c_SS_maxsize/3) ) &&
		( merged[iAxis] > (orig1[iAxis]+1) ) && 
		( merged[iAxis] > (orig2[iAxis]+1) ) )
		return FALSE;
	else
		return TRUE;
}



IC BOOL	ValidateMerge	(u32 f1, const Fbox& bb_base, const Fbox& bb_base_orig, u32 f2, const Fbox& bb, float& volume)
{
	
	// Polygons
	if ((f1+f2) > u32(4*c_SS_HighVertLimit/3))		return FALSE;	// Don't exceed limits (4/3 max POLY)	

 
	// Size
	Fbox	merge;
	merge.merge		(bb_base,bb);

	Fvector sz, orig1, orig2;
	merge.getsize	(sz);		
	bb_base_orig.getsize(orig1);
	bb.getsize		(orig2);
 
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 0))	return FALSE;	// Don't exceed limits (4/3 GEOM)
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 1))	return FALSE;
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 2))	return FALSE;

	// Volume
	Fbox		bb0,bb1;
	MakeCube	(bb0,bb_base);
	float	v1	= bb0.getvolume	();  
	MakeCube	(bb1,bb);	
	float	v2	= bb1.getvolume	();

	volume		= merge.getvolume	(); 

		
	
 	if (volume > 8 * ( v1 + v2)) // 2 * 2 * 2		
		return FALSE;	// Don't merge too distant groups (8 vol)
 

	// OK
	return TRUE;
}
 
// MERGING GEOMETRY FUNCTIONS

ICF void FindBestMergeCandidate(u32* selected ,  float* selected_volume , u32 split , u32 split_size , Fbox* bb_base_orig , Fbox* bb_base )
{
	int CUR_ID = *selected;

	
 	for ( u32 test = split ; test < split_size ; test++ ) 
	{	
		if (g_XSplit[test]->empty())
			continue;

		Fbox bb;
		float volume;
		vecFace& TEST = *( g_XSplit[test] );
		vecFace* subdiv = (g_XSplit[CUR_ID]);

 		if ( ! FaceEqual( subdiv->front() , TEST.front() ) )
			continue;
 		if ( ! NeedMerge( TEST , bb, test) )
			continue;
		if ( ! ValidateMerge( subdiv->size() , *bb_base , *bb_base_orig , TEST.size() , bb , volume ) )
			continue;
		
 		if ( volume < *selected_volume)
		{
			*selected = test;
			*selected_volume	= volume;			
 		}
	}
}

struct data_vec
{
	int face_id;
	bool merged = false;
};

struct data_faces
{
	xr_vector<data_vec> faces_vec;

};

xr_map<int, data_faces> thread_faces;
 
#include <execution>

// TH, MERGED
 
CTimer tGlobalMerge; 

IC void FindBestMergeCandidateTH(bool USE_MT, int ID, u32* selected, float* selected_volume, Fbox* bb_base_orig, Fbox* bb_base, xr_vector<data_vec>::iterator& vec_it)
{
	int CUR_ID = *selected;
	 
   	float volume;


	if (thread_faces[ID].faces_vec.size() > 24000 && USE_MT)
	{
		auto it = std::find_if(std::execution::par, thread_faces[ID].faces_vec.begin(), thread_faces[ID].faces_vec.end(), 
		[&] (data_vec& test)
		{
			//if (!g_XSplit[test.face_id])
			//	return false;

		  	//if (g_XSplit[test.face_id]->empty())
			//	return false;

			if (test.merged)
				return false;

			Fbox bb;
 			vecFace& TEST = *(g_XSplit[test.face_id]);
			vecFace* subdiv = (g_XSplit[CUR_ID]);
 
 			if (!FaceEqual(subdiv->front(), TEST.front()))
				return false;

  			if (!NeedMerge(TEST, bb, test.face_id))
				return false;
 
			if (!ValidateMerge(subdiv->size(), *bb_base, *bb_base_orig, TEST.size(), bb, volume))
				return false;
			
			if (volume < *selected_volume)
				return true;
 
			return false;
		});
		
		if (it != thread_faces[ID].faces_vec.end())
		{
			vec_it = it;
			*selected = (*it).face_id;
			*selected_volume = volume;
		}
	}
	else 
	{
		//for (auto test : thread_faces[ID].faces_vec)
		auto it = std::find_if(thread_faces[ID].faces_vec.begin(), thread_faces[ID].faces_vec.end(), 
		[&] (data_vec test)
		{ 
			if (test.merged)
				return false; 		 

			Fbox bb;
			float volume;
			vecFace& TEST = *(g_XSplit[test.face_id]);
			vecFace* subdiv = (g_XSplit[CUR_ID]);

 			if (!FaceEqual(subdiv->front(), TEST.front()))
				return false;

  			if (!NeedMerge(TEST, bb, test.face_id))
				return false;
 
			if (!ValidateMerge(subdiv->size(), *bb_base, *bb_base_orig, TEST.size(), bb, volume))
				return false;
	 			
			if (volume < *selected_volume)
				return true;

			return false;

			/* 
 			if (volume < *selected_volume)
			{
			
				*selected = test.face_id;
				*selected_volume = volume;						
				break;
			}
			*/
		});

		if (it != thread_faces[ID].faces_vec.end())
		{
			vec_it = it;
			*selected = (*it).face_id;
			*selected_volume = volume;
		}
	}

}

std::atomic<bool> stopped = false;

ICF void FindWhileMergeNeed(bool USE_MT, int id)
{ 
 
	for (;;)
	{
		xr_vector<data_vec>& faces_vect = thread_faces[id].faces_vec;

		lock.lock();
		if (faces_vect.size() == 0)
		{
			lock.unlock();
			break;
		}

		auto s = faces_vect.back().face_id;
		faces_vect.pop_back();		
		lock.unlock();
 
		if (g_XSplit[s]->empty())
			continue;

		Progress(1.0f / faces_vect.size());

		//StatusNoMSG("IDX: %d", faces_vect.size());

		vecFace& subdiv = *(g_XSplit[s]);
		bool		bb_base_orig_inited = false;
		Fbox		bb_base_orig;
		Fbox		bb_base;
 
		xr_vector<data_vec>::iterator vec_iter;
	
		int IDwhile = 0;
		while (NeedMerge(subdiv, bb_base, s))
		{
			StatusNoMSG("IDX_Works[%d] MERGE: %d TIMESEC: [%d], TRY_FIND: %d", thread_faces.size(), faces_vect.size(), (u32) tGlobalMerge.GetElapsed_sec(), IDwhile);
			IDwhile++;
 

			//	Save original AABB for later tests
			if (!bb_base_orig_inited)
			{
				bb_base_orig_inited = true;
				bb_base_orig = bb_base;
			}

			// **OK**. Let's find the best candidate for merge
			u32	selected = s;
			float	selected_volume = flt_max;

			
			FindBestMergeCandidateTH(USE_MT, id, &selected, &selected_volume, &bb_base_orig, &bb_base, vec_iter);
			
			if (selected == s)
				break;

			// **OK**. Perform merge
			lock.lock();
			subdiv.insert(subdiv.begin(), g_XSplit[selected]->begin(), g_XSplit[selected]->end());
			g_XSplit[selected]->clear_not_free();
 			bb_bases.erase(s);
			(*vec_iter).merged = true;  
 			lock.unlock();
		}

		// REMOVE MERGED
		{
		   	lock.lock();
			xr_vector<data_vec>& vec = thread_faces[id].faces_vec; 
			vec.erase(std::remove_if(vec.begin(), vec.end(), [] (data_vec& vec){return vec.merged;}), vec.end()); 
			lock.unlock();
		} 


	}
}

// Thread Pools
xr_vector<int> reserved;
xr_vector<int> reserved_big_objects;
 
 
#define MAX_BIG_THREADS 2

#include "../XrLCLight/BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;


void CBuild::xrPhase_MergeGeometry()
{
	Status("Processing...");
	validate_splits();

	tGlobalMerge.Start();

 	 
	CTimer t; 
	t.Start();

 	{ 
  		for (int split = 0; split < g_XSplit.size(); split++)
 			thread_faces[g_XSplit[split]->front()->dwMaterial].faces_vec.push_back(data_vec{split});
 
  
		for (auto mat : thread_faces)
		{
			if (mat.second.faces_vec.size() < 40000)
				reserved.push_back(mat.first);
			else 
				reserved_big_objects.push_back(mat.first);
		}
		// Run Work
		//FindSelectedMaterialCandidate();
 
		{
			

			std::thread* th = new std::thread[build_args->use_threads];
			for (auto i = 0; i < build_args->use_threads; i++)
			{	
				th[i] =	std::thread([&] ()
				{
						u32 last_ms = 0;
						for (;;)
						{
							lock.lock();
							if (reserved.empty())
							{
								lock.unlock();
								break;
							}
		 
 
							auto id = reserved.back();
							clMsg("Merge candidates:%d, Reserved: %d, PrevCalc: %u ms", thread_faces[id].faces_vec.size(),  reserved.size(), last_ms );		
							Progress( float ( 1 / reserved.size()) );
							reserved.pop_back();
							lock.unlock();
		
							CTimer t;t.Start();		  
							FindWhileMergeNeed(false, id);
							last_ms = t.GetElapsed_ms();
						}
				} 
				);
			}


			for (auto i = 0; i < build_args->use_threads; i++)
				th[i].join();
		
		}


		
		{
			std::thread* th = new std::thread[MAX_BIG_THREADS];
			for (auto i = 0; i < MAX_BIG_THREADS; i++)
				th[i] = std::thread([&] () 
			{
				u32 last_ms = 0;
				for (;;)
				{
					lock.lock();
					if (reserved_big_objects.empty())
					{
						lock.unlock();
						break;
					}
 
					auto id = reserved_big_objects.back();
					clMsg("Merge BIG candidates:%d, Reserved: %d, PrevCalc: %u ms", thread_faces[id].faces_vec.size(),  reserved_big_objects.size(), last_ms );		
					Progress( float ( 1 / reserved_big_objects.size()) );
					reserved_big_objects.pop_back();
					lock.unlock();
		
					CTimer t;t.Start();
					FindWhileMergeNeed(true, id);
					last_ms = t.GetElapsed_ms();
				}
			});

			for (auto i = 0; i < MAX_BIG_THREADS; i++)
				th[i].join();
		}

		// Clear Data
		thread_faces.clear();
		
		g_XSplit.erase(std::remove_if(g_XSplit.begin(), g_XSplit.end(), [](vecFace* ptr) { if (ptr == nullptr) return true; return ptr->empty(); }), g_XSplit.end());

	}
 
	
	clMsg("%d subdivisions.",g_XSplit.size());
	validate_splits			();

}


