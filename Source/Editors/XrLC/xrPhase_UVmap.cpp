#include "StdAfx.h"
#include "Build.h"

#include "../xrLCLight/xrDeflector.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrFace.h"

void Detach(xr_vector<Face*>* S)
{
 	xr_map<Vertex*, Vertex*> verts;
	verts.clear();

	// Collect vertices
	for (auto F = S->begin(); F != S->end(); ++F)
	{
		for (int i = 0; i < 3; ++i)
		{
			Vertex* V = (*F)->v[i];
			Vertex* VC;
			auto	W = verts.find(V);	// iterator

			if (W == verts.end())
			{	// where is no such-vertex
				VC = V->CreateCopy_NOADJ(lc_global_data()->g_vertices());	// make copy
				verts.insert(std::make_pair(V, VC));
			}
			else
			{
				// such vertex(key) already exists - update its adjacency
				VC = W->second;
			}
			VC->prep_add(*F);
			V->prep_remove(*F);
			(*F)->v[i] = VC;
		}
	}
	// vertices are already registered in container
	// so we doesn't need "vers" for this time
	verts.clear();
}

bool sort_faces(Face* face, Face* face2)
{
	if (face->CalcArea() > face2->CalcArea())
		return true;
	return false;
}

void CBuild::xrPhase_UVmap()
{
	size_t used, rel, free;
	vminfo(&free, &rel, &used);

	clMsg("xrPhase_UVmap: Start %u used", size_t(used / 1024 / 1024));

	// Main loop
	Status("Processing...");
	lc_global_data()->g_deflectors().reserve(64 * 1024);
	float		p_cost = 1.f / float(g_XSplit.size());
	float		p_total = 0.f;
	xr_vector<Face*>		faces_affected;

	int StartPoint = g_XSplit.size();
  	for (int SP = 0; SP < int(StartPoint); SP++)
	{
		Progress(p_total += p_cost);

		// Detect vertex-lighting and avoid this subdivision
		if (g_XSplit[SP]->empty())				continue;
		Face* Fvl = g_XSplit[SP]->front();
		if (Fvl->Shader().flags.bLIGHT_Vertex) 	continue;	// do-not touch (skip)
		if (!Fvl->Shader().flags.bRendering) 	continue;	// do-not touch (skip)
		if (Fvl->hasImplicitLighting())			continue;	// do-not touch (skip)

		while (TRUE)
		{
			// Сортировка списка в перед с больщими зонами.
			std::sort(g_XSplit[SP]->begin(), g_XSplit[SP]->end(), sort_faces);
			if (g_XSplit[SP] == nullptr)	break;
			
			// Select maximal sized poly
			Face* msF = NULL;

			for (auto FACE : *g_XSplit[SP])
			{
				if (FACE && FACE->pDeflector == nullptr)
				{
					msF = FACE;

					CDeflector* D = new CDeflector();
					lc_global_data()->g_deflectors().push_back(D);
					
					// Start recursion from this face
					start_unwarp_recursion();
					D->OA_SetNormal(FACE->N);

					faces_affected.clear();
					FACE->OA_Unwarp(D, faces_affected);

					// break the cycle to startup again
					D->OA_Export();

					// detaching itself
					Detach(&faces_affected);
					g_XSplit.push_back(new xr_vector<Face*>(faces_affected));
 				}
			}

			if (!g_XSplit[SP]->empty())
			{
 				auto rIT = std::remove_if(
					g_XSplit[SP]->begin(),
					g_XSplit[SP]->end(),
					[&](Face* F)
					{
						if (F->pDeflector != nullptr)
  							return true;    // Убираем из контейнера
 						return false;
					}
				);

				if (rIT != g_XSplit[SP]->end())
				{
					g_XSplit[SP]->erase(rIT, g_XSplit[SP]->end());
					g_XSplit[SP]->shrink_to_fit();
				}
			}

			// Cancel infine loop (while)
			if (msF == nullptr)		break;
		}

		AditionalData("SP[%u], xsp: %u", SP, g_XSplit.size() );
	}

  	clMsg("%d subdivisions...", g_XSplit.size());
	err_save();
}
