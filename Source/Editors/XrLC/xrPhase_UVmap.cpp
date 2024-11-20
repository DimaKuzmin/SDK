#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrDeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"

#include <mutex>
 
std::mutex mtx; 
void Detach(vecFace* S)
{
	map_v2v			verts;
	verts.clear		();
	
	// Collect vertices
	for (vecFaceIt F=S->begin(); F!=S->end(); ++F)
	{
		for (int i=0; i<3; ++i) 
		{
			Vertex*		V=(*F)->v[i];
			Vertex*		VC;
			map_v2v_it	W=verts.find(V);	// iterator
 
			if (W==verts.end()) 
			{	// where is no such-vertex
				
				// std::lock_guard lock(mtx);
				VC = V->CreateCopy_NOADJ( lc_global_data()->g_vertices() );	// make copy
				verts.insert(mk_pair(V, VC));
 			}
			else 
			{
				// such vertex(key) already exists - update its adjacency
				VC = W->second;
			}
			VC->prep_add		(*F);
			V->prep_remove		(*F);
			(*F)->v[i]=VC;
		}
	}

 	// vertices are already registered in container
	// so we doesn't need "vers" for this time
	verts.clear	();
}
 
bool sort_faces(Face* face, Face* face2)
{
	if (face->CalcArea() > face2->CalcArea())
		return true;
	return false;
}

#include <execution>

void CBuild::xrPhase_UVmap()
{
	// Main loop
	Status					("Processing...");
	lc_global_data()->g_deflectors().reserve	(64*1024);
	float		p_cost	= 1.f / float(g_XSplit.size());
	float		p_total	= 0.f;


	vecFace		faces_affected;
	clMsg("SP_SIZE %d, pixel_per_metter %f, Jitter = %d", g_XSplit.size(), g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);
   	u64 LastXSplit = g_XSplit.size();

	for (int SP = 0; SP < int(g_XSplit.size()); SP++) 
	{
		Progress			(p_total+=p_cost);
 
		// Detect vertex-lighting and avoid this subdivision
		R_ASSERT	(!g_XSplit[SP]->empty());
		Face*		Fvl = g_XSplit[SP]->front();
		if (Fvl->Shader().flags.bLIGHT_Vertex) 
			continue;	// do-not touch (skip)
		if (!Fvl->Shader().flags.bRendering) 	
			continue;	// do-not touch (skip)

		if (Fvl->hasImplicitLighting())		
			continue;	// do-not touch (skip)
 
		//   find first poly that doesn't has mapping and start recursion

 		vecFace* faces_selected = g_XSplit[SP];
		// vecFaceIt last_checked_id = faces_selected->begin();

		while (TRUE) 
		{  
			if (LastXSplit < SP)
				break;

			Face* msFaceLast = nullptr;
		
			for ( auto pIT = faces_selected->begin(); pIT < faces_selected->end(); pIT++)
			{
				Face* msF = *pIT;
				if (msF && msF->pDeflector == nullptr)
				{
					msFaceLast = msF;

					CDeflector* D = xr_new<CDeflector>();
					lc_global_data()->g_deflectors().push_back(D);

					faces_affected.clear();

					// Start recursion from this face
					start_unwarp_recursion();
					D->OA_SetNormal(msF->N);
					msF->OA_Unwarp(D, faces_affected);

					// break the cycle to startup again
					D->OA_Export();

					// Detach affected faces
					Detach(&faces_affected);
					g_XSplit.push_back(xr_new<vecFace>(faces_affected));
					StatusNoMSG("SP[%d], faces[%d], all[%d]", SP, LastXSplit, g_XSplit.size());
				}
			}
 
			//Status("Check SP[%d], IsEmpty(%d), Size: (%d), TOTAL(%d)", SP, g_XSplit[SP]->empty(), g_XSplit[SP]->size(), g_XSplit.size());

 			g_XSplit[SP]->erase(
				std::remove_if(
					g_XSplit[SP]->begin(),
					g_XSplit[SP]->end(), [](Face* F) 
					{ 
						return F->pDeflector != nullptr; 
					}
				) // remove_if
				, g_XSplit[SP]->end()
			);
 
			// Status("SP[%d], IsEmpty(%d), Size: (%d), TOTAL(%d)", SP, g_XSplit[SP]->empty(), g_XSplit[SP]->size(), g_XSplit.size());
			if (g_XSplit[SP]->empty())
			{
				xr_delete(g_XSplit[SP]);
				g_XSplit.erase(g_XSplit.begin() + SP);
				SP++;
			}
			

			if (msFaceLast == nullptr)
				break; // While Cycle Stop (NO finded Affected !!!)

		}	
	}

	g_XSplit.erase(std::remove_if(g_XSplit.begin(), g_XSplit.end(), [](vecFace* ptr) { return ptr->empty(); }), g_XSplit.end());
 
	clMsg("%d subdivisions...",g_XSplit.size());
	err_save		();
}

void CBuild::mem_CompactSubdivs()
{
	// Memory compact
	CTimer	dwT;	dwT.Start();
	vecFace			temp;
	for (int SP = 0; SP<int(g_XSplit.size()); SP++) 
	{
		temp.clear			();
		temp.assign			(g_XSplit[SP]->begin(),g_XSplit[SP]->end());
		xr_delete			(g_XSplit[SP]);
		mem_Compact			();
		g_XSplit[SP]		= xr_new<vecFace> ();
		g_XSplit[SP]->assign(temp.begin(),temp.end());
	}
	clMsg		("%d ms for memory compacting...",dwT.GetElapsed_ms());
}
void CBuild::mem_Compact()
{
	Msg("Start Memory Compact");
	log_vminfo();
	Memory.mem_compact	();
	log_vminfo();
	Msg("End Memory Compact");

	/*
	u32					bytes,blocks_used,blocks_free;
	bytes				= Memory.mem_usage(&blocks_used,&blocks_free);
	LPCSTR h_status		= 0;
	if (HeapValidate	(GetProcessHeap(),0,0))	h_status = "OK";
	else										h_status = "DAMAGED";
	clMsg				("::MEMORY(%s):: %d MB, %d Bused, %d Bfree",
		h_status,bytes/(1024*1024),blocks_used,blocks_free);
	*/
}
