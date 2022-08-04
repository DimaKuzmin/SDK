#include "stdafx.h"

#include "build.h"
#include "../xrLCLight/xrface.h"

#include <thread>

extern void Detach		(vecFace* S);

IC BOOL	FaceEqual		(Face* F1, Face* F2)
{
	if (F1->dwMaterial  != F2->dwMaterial)		return FALSE;
	if (F1->tc.size()	!= F2->tc.size())		return FALSE;
	if (F1->lmap_layer  != F2->lmap_layer)		return FALSE;
	return TRUE;
}

Fbox CreateBbox(vecFace& subdiv)
{
	Fbox bb_base;

	bb_base.invalidate();
	for (u32 it = 0; it < subdiv.size(); it++)
	{
		Face* F = subdiv[it];
		bb_base.modify(F->v[0]->P);
		bb_base.modify(F->v[1]->P);
		bb_base.modify(F->v[2]->P);
	}
	bb_base.grow(EPS_S);	// Enshure non-zero volume

	return bb_base;
}

BOOL	NeedMerge		(vecFace& subdiv, Fbox& bb_base)
{
	// 1. Amount of polygons
	if (subdiv.size()>=u32(3*c_SS_HighVertLimit/4))	return FALSE;
	
	// 2. Bounding box
	bb_base.invalidate	();
	for (u32 it=0; it<subdiv.size(); it++)
	{
		Face* F = subdiv[it];
		bb_base.modify(F->v[0]->P);
		bb_base.modify(F->v[1]->P);
		bb_base.modify(F->v[2]->P);
	}
	bb_base.grow		(EPS_S);	// Enshure non-zero volume

	Fvector sz_base;	bb_base.getsize(sz_base);
	if (sz_base.x<c_SS_maxsize)		return TRUE;
	if (sz_base.y<c_SS_maxsize)		return TRUE;
	if (sz_base.z<c_SS_maxsize)		return TRUE;
	return FALSE;
}

BOOL	NeedMergeBBOX(vecFace& subdiv, Fbox& bb_base)
{
	// 1. Amount of polygons
	if (subdiv.size() >= u32(3 * c_SS_HighVertLimit / 4))	return FALSE;

	Fvector sz_base;	bb_base.getsize(sz_base);
	if (sz_base.x < c_SS_maxsize)		return TRUE;
	if (sz_base.y < c_SS_maxsize)		return TRUE;
	if (sz_base.z < c_SS_maxsize)		return TRUE;
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
	volume		= merge.getvolume	(); // / Cuboid(merge);
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
	bool fast = strstr(Core.Params, "-fast_merge_geom");
	while( TRUE ) {
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

xr_vector<int> FindBestMergeCandidate_threads( u32* selected ,  float* selected_volume , u32 split , u32 split_size , vecFace* subdiv , Fbox* bb_base_orig , Fbox* bb_base )
{
	u32 m_range = ( split_size - split ) / mergegm_threads_count;

	// Assigning parameters
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ ) {
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

	/*
	// Compose results
	for ( u32 i = 0 ; i < mergegm_threads_count ; i++ ) 
	{
		if ( mergegm_params[ i ].selected_volume < *selected_volume ) 
		{
			*selected = mergegm_params[ i ].selected;
			*selected_volume = mergegm_params[ i ].selected_volume;
		}
	}
	*/

	xr_vector<int> th_returns;

	for (u32 i = 0; i < mergegm_threads_count; i++)
	{
		if (mergegm_params[i].selected != *selected)
			th_returns.push_back(mergegm_params[i].selected);
	}

	return th_returns;
	
}

u64 ticks_FACE_EQ = 0;
u64 ticks_FACE_NEED = 0;
u64 ticks_FACE_VALID = 0;

IC void FindBestMergeCandidate(u32* selected ,  float* selected_volume , u32 split , u32 split_size , Fbox* bb_base_orig , Fbox* bb_base )
{
	int CUR_ID = *selected;

	CTimer t;
	for ( u32 test = split ; test < split_size ; test++ ) 
	{	
		if (g_XSplit[test]->empty())
			continue;

		Fbox bb;
		float volume;
		vecFace& TEST = *( g_XSplit[test] );
		vecFace* subdiv = (g_XSplit[CUR_ID]);

		t.Start();
		if ( ! FaceEqual( subdiv->front() , TEST.front() ) )
			continue;
		ticks_FACE_EQ += t.GetElapsed_ticks();
		if ( ! NeedMerge( TEST , bb ) )
			continue;
		ticks_FACE_NEED += t.GetElapsed_ticks();
		
		if ( ! ValidateMerge( subdiv->size() , *bb_base , *bb_base_orig , TEST.size() , bb , volume ) )
			continue;
		
		ticks_FACE_VALID += t.GetElapsed_ticks();
		if ( volume < *selected_volume)
		{
			*selected = test;
			*selected_volume	= volume;
			 
			break;
 		}
	}
}

xr_map<int, xr_vector<int>> thread_faces;

#include <mutex>

xrCriticalSection cs;
std::mutex lock;
int ready_threads;

IC void FindBestMergeCandidateTH(int th_ID, u32* selected, float* selected_volume, u32 split, u32 split_size, Fbox* bb_base_orig, Fbox* bb_base)
{
	int CUR_ID = *selected;
	CTimer t;
	for (auto test : thread_faces[th_ID])
	{
		if (g_XSplit[test]->empty())
			continue;

		Fbox bb;
		float volume;
		vecFace& TEST = *(g_XSplit[test]);
		vecFace* subdiv = (g_XSplit[CUR_ID]);

		t.Start();
		if (!FaceEqual(subdiv->front(), TEST.front()))
			continue;
		ticks_FACE_EQ += t.GetElapsed_ticks();
		if (!NeedMerge(TEST, bb))
			continue;
		ticks_FACE_NEED += t.GetElapsed_ticks();

		if (!ValidateMerge(subdiv->size(), *bb_base, *bb_base_orig, TEST.size(), bb, volume))
			continue;

		ticks_FACE_VALID += t.GetElapsed_ticks();
		if (volume < *selected_volume)
		{
			*selected = test;
			*selected_volume = volume;

			break;
		}
	}
}

IC void FindSelectedMaterialCandidate(int th_id)
{
	clMsg("RunThread %d", th_id);
	int ids = 0;
	for (auto s : thread_faces[th_id])
	{
		ids++;
		if (ids % 256 == 0)
		{
			clMsg("Merge th_id: %d, progress %d/%d", th_id, ids, thread_faces[th_id].size());
		}  		 

		if (g_XSplit[s]->empty())
			continue;
		//clMsg("Merge ID: %d, th %d", s, th_id);
		vecFace& subdiv = *(g_XSplit[s]);
		bool		bb_base_orig_inited = false;
		Fbox		bb_base_orig;
		Fbox		bb_base;

		while (NeedMerge(subdiv, bb_base))
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

			FindBestMergeCandidateTH(th_id, &selected, &selected_volume, s + 1, g_XSplit.size(), &bb_base_orig, &bb_base);

			if (selected == s)
				break;

			// **OK**. Perform merge
			lock.lock();
			 
			subdiv.insert(subdiv.begin(), g_XSplit[selected]->begin(), g_XSplit[selected]->end());
			//g_XSplit[selected] = 0;
			//erased[selected] = true;
			g_XSplit[selected]->clear_not_free();
			 
			lock.unlock();
			
			//xr_delete(g_XSplit[selected]);
			//g_XSplit.erase(g_XSplit.begin() + selected);
			 
		}
	}
	clMsg("Ready TH %d", th_id);
	ready_threads++;
}

#define MAX_THREADS 8

void CBuild::xrPhase_MergeGeometry()
{
	Status("Processing...");
	validate_splits();

	// Initialize helper threads
//	InitMergeGmThreads();

	bool use_fast_way = strstr(Core.Params, "-merge_fast");

	if (use_fast_way)
	{
		xr_map<int, xr_vector<int>> materialfaces;
 
		int SIZE_FACES = 0;
		for (int split = 0; split < g_XSplit.size(); split++)
			SIZE_FACES += g_XSplit[split]->size();
		int slice_part = SIZE_FACES / MAX_THREADS;

		int cur_size = 0;
		int th_num = 1;

		for (int split = 0; split < g_XSplit.size(); split++)
			materialfaces[g_XSplit[split]->front()->dwMaterial].push_back(split);

		for (int split = 0; split < materialfaces.size(); split++)
		{
			if (cur_size > slice_part)
			{
				cur_size = 0;
				th_num++;
			}

			for (auto id : materialfaces[split])
			{
				cur_size += g_XSplit[id]->size();
				thread_faces[th_num].push_back(id);
			}
		}

		materialfaces.clear();

		for (int i = 1; i <= MAX_THREADS; i++)
		{
			std::thread* th = new std::thread(FindSelectedMaterialCandidate, i);
			th->detach();
		}

		while (ready_threads < MAX_THREADS)
		{
			clMsg("Ready THs: %d/%d", ready_threads, MAX_THREADS);
			Sleep(1000);
		}

		thread_faces.clear();

		for (int split = 0; split < g_XSplit.size(); split++)
		{
 			StatusNoMSG("Progress ERASE: %d\%d", split, g_XSplit.size());
			if (g_XSplit[split]->empty())
			{
				g_XSplit.erase(g_XSplit.begin() + split);
				if (split > 0)
					split--;
			}
		}
	}
	else
	{ 
		for (u32 split = 0; split < g_XSplit.size(); split++)
		{
			vecFace& subdiv = *(g_XSplit[split]);
			bool		bb_base_orig_inited = false;
			Fbox		bb_base_orig;
			Fbox		bb_base;

			while (NeedMerge(subdiv, bb_base))
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

				FindBestMergeCandidate(&selected, &selected_volume, split + 1, g_XSplit.size(), &bb_base_orig, &bb_base);

				if (selected == split)
					break;

				// **OK**. Perform merge
				subdiv.insert(subdiv.begin(), g_XSplit[selected]->begin(), g_XSplit[selected]->end());
				xr_delete(g_XSplit[selected]);
				g_XSplit.erase(g_XSplit.begin() + selected);
			}

			Progress(float(split) / float(g_XSplit.size()));
			StatusNoMSG("Merge %d/%d", split, g_XSplit.size());
		}
	}

	// Destroy helper threads
	//DoneMergeGmThreads();
	
	clMsg("%d subdivisions.",g_XSplit.size());
	validate_splits			();

}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    