#include "stdafx.h"

#include "build.h"
#include "../xrLCLight/xrface.h"

#include <thread>

extern void Detach		(vecFace* S);

IC BOOL	FaceEqual		(Face* F1, Face* F2)
{
	if (F1->v[0]->P.distance_to(F2->v[0]->P) > 256 ) return FALSE;
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
	
	// 2. Bounding box
	bb_base.invalidate	();
	CreateBox(subdiv, bb_base, id);
	
	bb_base.grow		(EPS_S);	// Enshure non-zero volume
	Fvector sz_base;
	bb_base.getsize(sz_base);

	
	if (sz_base.x<c_SS_maxsize)		return TRUE;
	if (sz_base.y<c_SS_maxsize)		return TRUE;
	if (sz_base.z<c_SS_maxsize)		return TRUE;
	return FALSE;
}
 
float	Cuboid			(Fbox& BB)
{
	Fvector sz;			BB.getsize(sz);
	float	min			= sz.x;
	if (sz.y<min)	min = sz.y;
	if (sz.z<min)	min = sz.z;
	
	float	volume_cube	= min*min*min;
	float	volume		= sz.x*sz.y*sz.z;
	return  powf(volume_cube / volume, 1.f/7.f);
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
	Fbox	merge;	merge.merge		(bb_base,bb);
	Fvector sz;		merge.getsize	(sz);
	Fvector orig1;	bb_base_orig.getsize(orig1);
	Fvector orig2;	bb.getsize		(orig2);
 
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 0))	return FALSE;	// Don't exceed limits (4/3 GEOM)
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 1))	return FALSE;
	if (!ValidateMergeLinearSize(sz, orig1, orig2, 2))	return FALSE;

	// Volume
	Fbox		bb0,bb1;
	MakeCube	(bb0,bb_base);	float	v1	= bb0.getvolume	();
	MakeCube	(bb1,bb);		float	v2	= bb1.getvolume	();
	volume		= merge.getvolume	(); 
	
	//Cuboid(merge);
	if (volume > 2*2*2*(v1+v2))						return FALSE;	// Don't merge too distant groups (8 vol)

	// OK
	return TRUE;
}

void FindBestMergeCandidate( u32* selected ,  float* selected_volume , u32 split , u32 split_size , Fbox* bb_base_orig , Fbox* bb_base );
//void FindMergeBBOX(u32* selected, float* selected_volume, u32 split, u32 split_size, vecFace* subdiv, Fbox* bb_base_orig, Fbox* bb_base);

typedef struct MERGEGM_PARAMS
{
	u32 selected;
	float selected_volume;
	u32 split;
	u32 split_size;
	vecFace* subdiv;
	Fbox* bb_base_orig;
	Fbox* bb_base;
	HANDLE hEvents[3];	// 0=start,1=terminate,2=ready
} * LP_MERGEGM_PARAMS;

static CRITICAL_SECTION mergegm_cs;
static BOOL mergegm_threads_initialized = FALSE;
static u32 mergegm_threads_count = 0;
static LPHANDLE mergegm_threads_handles = NULL;
static LPHANDLE mergegm_ready_events = NULL;
static LP_MERGEGM_PARAMS mergegm_params = NULL;

DWORD WINAPI MergeGmThreadProc( LPVOID lpParameter )
{
	LP_MERGEGM_PARAMS pParams = ( LP_MERGEGM_PARAMS ) lpParameter;
 	while( TRUE ) 
	{
		// Wait for "start" and "terminate" events
		switch ( WaitForMultipleObjects( 2 , pParams->hEvents , FALSE , INFINITE ) ) {
			case WAIT_OBJECT_0 + 0 :				 
				FindBestMergeCandidate(
					&pParams->selected, &pParams->selected_volume, pParams->split, pParams->split_size,
					 pParams->bb_base_orig, pParams->bb_base
				);
 

				// Signal "ready" event
				SetEvent( pParams->hEvents[ 2 ] );
				break;
			case WAIT_OBJECT_0 + 1 :
				ExitThread( 0 );
				break;
			default :
				// Error ?
				ExitThread( 1 );
				break;
		} // switch
	} // while

	return 0;
}

void InitMergeGmThreads()
{
	if ( mergegm_threads_initialized )
		return;
	
	SYSTEM_INFO SystemInfo;
	GetSystemInfo( &SystemInfo );
	mergegm_threads_count = SystemInfo.dwNumberOfProcessors;

	mergegm_threads_handles = (LPHANDLE) xr_malloc( mergegm_threads_count * sizeof( HANDLE ) );
	mergegm_ready_events = (LPHANDLE) xr_malloc( mergegm_threads_count * sizeof( HANDLE ) );
	mergegm_params = (LP_MERGEGM_PARAMS) xr_malloc( mergegm_threads_count * sizeof( MERGEGM_PARAMS ) );

	InitializeCriticalSection( &mergegm_cs );

	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ ) {

		ZeroMemory( &mergegm_params[ i ] , sizeof( MERGEGM_PARAMS ) );

		// Creating start,terminate,ready events for each thread
		for( u32 x = 0 ; x < 3 ; x++ )
			mergegm_params[ i ].hEvents[ x ] = CreateEvent( NULL , FALSE , FALSE , NULL );

		// Duplicate ready event into array
		mergegm_ready_events[ i ] = mergegm_params[ i ].hEvents[ 2 ];

		mergegm_threads_handles[ i ] = CreateThread( NULL , 0 , &MergeGmThreadProc , &mergegm_params[ i ] , 0 , NULL );
	}

	mergegm_threads_initialized = TRUE;
}

void DoneMergeGmThreads()
{
	if ( ! mergegm_threads_initialized )
		return;

	// Asking helper threads to terminate
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ )
		SetEvent( mergegm_params[ i ].hEvents[ 1 ] );

	// Waiting threads for completion
	WaitForMultipleObjects( mergegm_threads_count , mergegm_threads_handles , TRUE , INFINITE );

	// Deleting events
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ )
		for( u32 x = 0 ; x < 3 ; x++ )
			CloseHandle( mergegm_params[ i ].hEvents[ x ] );

	// Freeing resources
	DeleteCriticalSection( &mergegm_cs );

	xr_free( mergegm_threads_handles );		mergegm_threads_handles = NULL;
	xr_free( mergegm_ready_events );		mergegm_ready_events = NULL;
	xr_free( mergegm_params );				mergegm_params = NULL;

	mergegm_threads_count = 0;

	mergegm_threads_initialized = FALSE;
}

ICF void FindBestMergeCandidate_threads( u32* selected ,  float* selected_volume , u32 split , u32 split_size , vecFace* subdiv , Fbox* bb_base_orig , Fbox* bb_base )
{
	u32 m_range = ( split_size - split ) / mergegm_threads_count;

	// Assigning parameters
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ ) 
	{
		mergegm_params[ i ].selected = *selected;
		mergegm_params[ i ].selected_volume = *selected_volume;

		mergegm_params[ i ].split = split + ( i * m_range );
		mergegm_params[ i ].split_size = ( i == ( mergegm_threads_count - 1 ) ) ? split_size : mergegm_params[ i ].split + m_range;

		mergegm_params[ i ].subdiv = subdiv;
		mergegm_params[ i ].bb_base_orig = bb_base_orig;
		mergegm_params[ i ].bb_base = bb_base;

		SetEvent( mergegm_params[ i ].hEvents[ 0 ] );
	} // for

	
	// Wait for result
	WaitForMultipleObjects( mergegm_threads_count , mergegm_ready_events , TRUE , INFINITE );
		   
	// Compose results
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ ) 
	{
		if ( mergegm_params[ i ].selected_volume < *selected_volume ) 
		{
			*selected = mergegm_params[ i ].selected;
			*selected_volume = mergegm_params[ i ].selected_volume;
		}
	}
}

xrCriticalSection cs;


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

xr_vector<int> reserved;

struct data_vec
{
	int face_id;
};

struct data_faces
{
	xr_vector<data_vec> faces_vec;
};

xr_map<int, data_faces> thread_faces;
int ready_threads;

#include <execution>

IC void FindBestMergeCandidateTH(int ID, u32* selected, float* selected_volume, Fbox* bb_base_orig, Fbox* bb_base)
{
	int CUR_ID = *selected;
 
	if (thread_faces[ID].faces_vec.size() > 1)
	{
		float volume;
		auto it = std::find_if(std::execution::par, thread_faces[ID].faces_vec.begin(), thread_faces[ID].faces_vec.end(), 
		[&] (data_vec test)
		{
			if (!g_XSplit[test.face_id])
				return false;

		  	if (g_XSplit[test.face_id]->empty())
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
			else 
				return false;
		});
		
		if (it != thread_faces[ID].faces_vec.end())
		{
			*selected = (*it).face_id;
			*selected_volume = volume;
		}

	}
	else 
	for (auto test : thread_faces[ID].faces_vec)
	{ 
		if (g_XSplit[test.face_id]->empty())
			continue;

		Fbox bb;
		float volume;
		vecFace& TEST = *(g_XSplit[test.face_id]);
		vecFace* subdiv = (g_XSplit[CUR_ID]);

 		if (!FaceEqual(subdiv->front(), TEST.front()))
			continue;

  		if (!NeedMerge(TEST, bb, test.face_id))
			continue;
 
		if (!ValidateMerge(subdiv->size(), *bb_base, *bb_base_orig, TEST.size(), bb, volume))
			continue;

 		if (volume < *selected_volume)
		{
			*selected = test.face_id;
			*selected_volume = volume;						
			break;
		}
	}
	
}	  

ICF void FindWhileMergeNeed(int id)
{ 
	for (;;)
	{
		lock.lock();
		if (thread_faces[id].faces_vec.size() == 0)
		{
			lock.unlock();
			break;
		}

		int s = thread_faces[id].faces_vec.back().face_id;
		thread_faces[id].faces_vec.pop_back();		
		lock.unlock();

		if (g_XSplit[s]->empty())
			continue;

		Progress(1.0f / thread_faces[id].faces_vec.size());

		StatusNoMSG("IDX: %d", thread_faces[id].faces_vec.size());

		vecFace& subdiv = *(g_XSplit[s]);
		bool		bb_base_orig_inited = false;
		Fbox		bb_base_orig;
		Fbox		bb_base;

		while (NeedMerge(subdiv, bb_base, s))
		{
			//	Save original AABB for later tests
			if (!bb_base_orig_inited)
			{
				bb_base_orig_inited = true;
				bb_base_orig = bb_base;
			}

			// **OK**. Let's find the best candidate for merge
			u32	selected = s;
			float	selected_volume = flt_max;


			FindBestMergeCandidateTH(id, &selected, &selected_volume, &bb_base_orig, &bb_base);
			
			if (selected == s)
				break;

			// **OK**. Perform merge
			lock.lock();
			subdiv.insert(subdiv.begin(), g_XSplit[selected]->begin(), g_XSplit[selected]->end());
			g_XSplit[selected]->clear_not_free();
 			bb_bases.erase(s);
 			lock.unlock();
			
		}
	}
}

IC void FindSelectedMaterialCandidate()
{
	int current_id = 0;
	int max_id = reserved.size();
//#define MAX_THREADS 8
	CTimer t; t.Start();
 
	for (;;)
	{
		current_id++;

		lock.lock();
		if (reserved.empty())
		{
			lock.unlock();
			break;
		}

		int id = reserved.back();
		reserved.pop_back();
		lock.unlock();
		
 		clMsg("Merge id: %d/%d, candidates:%d, Second: %.0f", current_id, max_id, thread_faces[id].faces_vec.size(), t.GetElapsed_sec() );		
		FindWhileMergeNeed(id);
	}

	ready_threads++;
}

int THREADS_COUNT(); 
#define MAX_THREADS THREADS_COUNT()
 

void CBuild::xrPhase_MergeGeometry()
{
	Status("Processing...");
	validate_splits();

	bool use_fast_way = true;

	CTimer t; 
	t.Start();

	if (use_fast_way)
	{ 
 		for (int split = 0; split < g_XSplit.size(); split++)
			thread_faces[g_XSplit[split]->front()->dwMaterial].faces_vec.push_back(data_vec{split});

		for (auto mat : thread_faces)
			reserved.push_back(mat.first);
 		
		// Run Work
		FindSelectedMaterialCandidate();
 
		// Clear Data
		thread_faces.clear();
		g_XSplit.erase(std::remove_if(g_XSplit.begin(), g_XSplit.end(), [](vecFace* ptr) { return ptr->empty(); }), g_XSplit.end());

	}
	else
	{ 
		InitMergeGmThreads();

		for (u32 split = 0; split < g_XSplit.size(); split++)
		{
			vecFace& subdiv = *(g_XSplit[split]);
			bool		bb_base_orig_inited = false;
			Fbox		bb_base_orig;
			Fbox		bb_base;

			while (NeedMerge(subdiv, bb_base, split))
			{
				//	Save original AABB for later tests
				if (!bb_base_orig_inited)
				{
					bb_base_orig_inited = true;
					bb_base_orig = bb_base;
				}

				// **OK**. Let's find the best candidate for merge
				u32	selected = split;
				float	selected_volume = flt_max;

				FindBestMergeCandidate_threads(&selected, &selected_volume, split + 1, g_XSplit.size(), &subdiv, &bb_base_orig, &bb_base);

				if (selected == split)
					break;

				// **OK**. Perform merge
				subdiv.insert(subdiv.begin(), g_XSplit[selected]->begin(), g_XSplit[selected]->end());
				xr_delete(g_XSplit[selected]);
				g_XSplit.erase(g_XSplit.begin() + selected);
				bb_bases.erase(split);
			}

			Progress(float(split) / float(g_XSplit.size()));
			StatusNoMSG("Merge %d/%d, time: %.0f", split, g_XSplit.size(), t.GetElapsed_sec());
		}

		DoneMergeGmThreads();
	}
	
	clMsg("%d subdivisions.",g_XSplit.size());
	validate_splits			();

}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    