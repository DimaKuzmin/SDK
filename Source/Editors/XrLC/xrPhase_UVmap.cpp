#include "stdafx.h"
#include "build.h"

#include "../xrLCLight/xrDeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"

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

extern string_path LEVEL_PATH = {0};  
extern int orig_size = 0;

bool find_AFFECTED(Face* face)
{
	return face->pDeflector != NULL;
}

bool sort_faces(Face* face, Face* face2)
{
	if (face->CalcArea() > face2->CalcArea())
		return true;
	return false;
}

#include <atomic>
std::atomic<int> integer_mt;

void MT_FindAttached(CDeflector* defl, vecFace& affected, int SP, int start, int end)
{
	for (int i = start; i != end; i++)
	{
		auto Face = (*g_XSplit[SP])[i];

		if (Face->pDeflector == defl)
		{
			affected.push_back(Face); 
			integer_mt.fetch_add(1);
		}	 
	}
}

#include <thread>
#include <algorithm>
#include <execution>

void CBuild::xrPhase_UVmap()
{
	// Main loop
	Status					("Processing...");
	lc_global_data()->g_deflectors().reserve	(64*1024);
	float		p_cost	= 1.f / float(g_XSplit.size());
	float		p_total	= 0.f;
	vecFace		faces_affected;
	u32			remove_count = 0;

	clMsg("SP_SIZE %d, pixel_per_metter %f, Jitter = %d", g_XSplit.size(), g_params().m_lm_pixels_per_meter, g_params().m_lm_jitter_samples);
 
	orig_size = g_XSplit.size();
	bool use_fast_method = true;  

	CTimer timer_gl, timer_c;
	timer_gl.Start();
 
	u64 ticks_new = 0;
	u64 ticks_buffer = 0;
	u64 ticks_find = 0;
	u64 ticks_find_affected = 0;
	u64 ticks_clear_finded = 0;
	
	u64 LastSP = 0;	
	u64 LastXSplit = g_XSplit.size();

	for (int SP = 0; SP<int(g_XSplit.size()); SP++) 
	{

		if (LastSP != SP && LastSP < LastXSplit)
		{
			LastSP = SP;

			int matterial  = g_XSplit[SP]->front()->dwMaterial;
			int surfaceID  = lc_global_data()->materials()[matterial].surfidx;
			int shaderID  = lc_global_data()->materials()[matterial].shader;

			Fvector pos;
			g_XSplit[SP]->front()->CalcCenter(pos);
 
			auto shader = lc_global_data()->shaders().Get(shaderID);
			auto texture = lc_global_data()->textures()[surfaceID];

			clMsg("SP[%d], mat: %d, surfaceID: %d, shaderID: %d, texture: %s, shader: %s", SP, matterial, surfaceID, shaderID, texture.name, shader->Name);
			clMsg("Position: {%f, %f, %f}", VPUSH(pos) );
		}

		Progress			(p_total+=p_cost);
 
		// ManOwaR, unsure:
		// Call to IsolateVertices() looks useless here
		// Calculation speed up, so commented
		// IsolateVertices		(FALSE);
		
		// Detect vertex-lighting and avoid this subdivision
		R_ASSERT	(!g_XSplit[SP]->empty());
		Face*		Fvl = g_XSplit[SP]->front();
		if (Fvl->Shader().flags.bLIGHT_Vertex) 	continue;	// do-not touch (skip)
		if (!Fvl->Shader().flags.bRendering) 	continue;	// do-not touch (skip)
		if (Fvl->hasImplicitLighting())			continue;	// do-not touch (skip)
 
		//   find first poly that doesn't has mapping and start recursion

		int id_face = 0;		
		vecFace* faces_selected = g_XSplit[SP];
		vecFaceIt last_checked_id = faces_selected->begin();

		if (use_fast_method)
		{
			remove_count = 0;
			std::sort(faces_selected->begin(), faces_selected->end(), sort_faces);
		}

		while (TRUE) 
		{ 
			// Select maximal sized poly
			Face *	msF		= NULL;
			float	msA		= 0;
			 
			timer_c.Start();
			for (vecFaceIt it = last_checked_id; it != faces_selected->end(); it++)
			{
				if ( (*it)->pDeflector == NULL )
				{	
 					msF = (*it);
					if (!use_fast_method)
					{
						float a = (*it)->CalcArea();
						if (a > msA)
						{
							msF = (*it);
							msA = a;
						}
					}
					else
					{
						id_face++;
						last_checked_id = it;
						break;
					}				
				}
			}
			ticks_find += timer_c.GetElapsed_ticks();

			 
			if (!msF && use_fast_method)
			{
 				if (remove_count == g_XSplit[SP]->size())
				{
					// clMsg("SP[%d], Removed: %d, size %d", SP, remove_count, g_XSplit[SP]->size());
					xr_delete(g_XSplit[SP]);
					g_XSplit.erase(g_XSplit.begin() + SP);
					SP--;
					break;
				}
				else if (remove_count > 0)
				{
					clMsg("CHECK ERROR SP[%d], Removed: %d, size %d", SP, remove_count, g_XSplit[SP]->size());
				}  
				else if (SP < orig_size)
				{
 				}
			}
		    
			if (msF)
			{
				timer_c.Start();
				CDeflector* D = xr_new<CDeflector>();
				lc_global_data()->g_deflectors().push_back(D);
			
 				faces_affected.clear();
 
				// Start recursion from this face
				start_unwarp_recursion();
				D->OA_SetNormal(msF->N);
				msF->OA_Unwarp(D, faces_affected);
				remove_count += faces_affected.size();
				
				// break the cycle to startup again
				D->OA_Export();
				ticks_new += timer_c.GetElapsed_ticks();

		
				// Detach affected faces
				// detaching itself
				
				timer_c.Start();
				Detach				(&faces_affected);
 				g_XSplit.push_back	(xr_new<vecFace> (faces_affected));

				ticks_buffer += timer_c.GetElapsed_ticks();
				StatusNoMSG("SP[%d], face[%d]/[%d], all[%d] T[%f]", SP, id_face, g_XSplit[SP]->size() - remove_count, g_XSplit.size(), timer_gl.GetElapsed_sec());
			}
			else
			{
				
				if (g_XSplit[SP]->empty() && SP >= 1)
				{
					xr_delete(g_XSplit[SP]);
					g_XSplit.erase(g_XSplit.begin() + SP);
					SP--;
				}
				

				// Cancel infine loop (while)
				StatusNoMSG("SP[%d], face[%d]/[%d], all[%d] T[%f]", SP, id_face, g_XSplit[SP]->size() - remove_count, g_XSplit.size(), timer_gl.GetElapsed_sec());

				//StatusNoMSG("SP[%d], face[%d]/[%d], all[%d]", SP, id_face, g_XSplit[SP]->size(), g_XSplit.size());
				break;
			}
				
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
