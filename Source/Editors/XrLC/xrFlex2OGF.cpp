#include "stdafx.h"
#include "build.h"
#include "OGF_Face.h"
#include "vbm.h"
//#include "std_classes.h"
#include "../xrLCLight/lightmap.h"
#include "../xrLCLight/xrface.h"

#define	TRY(a) try { a; } catch (...) { clMsg("* E: %s", #a); }

void CBuild::validate_splits			()
{
	int Errors = 0;
	for (splitIt it=g_XSplit.begin(); it!=g_XSplit.end(); it++)
	{
		u32 MODEL_ID		= u32(it-g_XSplit.begin())	;
 
		if ((*it)->size() > c_SS_HighVertLimit*2 || (*it)->size() == 0)
		{
			Errors++;
			clMsg	("! ERROR: subdiv #%d has more than %d faces (%d)",MODEL_ID,2*c_SS_HighVertLimit,(*it)->size());
		}
	};

	clMsg("[Subdivide Splits] Errors: %d", Errors);
}

void Face2OGF_Vertices( const Face &FF, OGF_Vertex	V[3] ) 
{
	for (u32 fv=0; fv<3; fv++)
	{
		V[fv].P.set	(FF.v[fv]->P);
		V[fv].N.set	(FF.v[fv]->N); 
		V[fv].T		= FF.basis_tangent[fv];
		V[fv].B		= FF.basis_binormal[fv];
		V[fv].Color	= FF.v[fv]->C;
	}
	
	// Normal order
	svector<_TCF,2>::const_iterator TC=FF.tc.begin(); 
	for (;TC!=FF.tc.end(); TC++)
	{
		V[0].UV.push_back(TC->uv[0]);
		V[1].UV.push_back(TC->uv[1]);
		V[2].UV.push_back(TC->uv[2]);
	}
}

void OGF_AddFace( OGF &ogf, const Face& FF, bool _tc_ )
{
	OGF_Vertex	V[3];
	// Geometry
	Face2OGF_Vertices( FF, V );
	// build face
	TRY				(ogf._BuildFace(V[0],V[1],V[2],_tc_));
	V[0].UV.clear();V[1].UV.clear();V[2].UV.clear();
}

void BuildOGFGeom( OGF &ogf, const vecFace& faces, bool _tc_ )
{
	for (vecFaceCit Fit=faces.begin(); Fit!=faces.end(); Fit++)
	{
		Face*	FF = *Fit;
		R_ASSERT(FF);
		OGF_AddFace( ogf, *FF, _tc_ );
	}
}
 
bool ConvertOgf(u32 THID, u32 MODEL_ID,  vecFace* faces , Face* F, b_material* M, OGF* pOGF, CBuild* build)
{
	try 
	{
 		// Common data
		pOGF->Sector = M->sector;
		pOGF->material = F->dwMaterial;

		// Collect textures
		OGF_Texture			T;
		TRY(T.name = build->textures()[M->surfidx].name);
		TRY(T.pBuildSurface = &(build->textures()[M->surfidx]));
		TRY(pOGF->textures.push_back(T));

		try 
		{
			if (F->hasImplicitLighting())
			{
				// specific lmap
				string_path		tn;
				strconcat(sizeof(tn), tn, *T.name, "_lm.dds");
				T.name = tn;
				T.pBuildSurface = T.pBuildSurface;	// Leave surface intact
				R_ASSERT(pOGF);
				pOGF->textures.push_back(T);
			}
			else
			{
				// If lightmaps persist
				CLightmap* LM = F->lmap_layer;
				if (LM)
				{
 					string_path	fn;
					xr_sprintf(fn, "%s_1", LM->lm_texture.name);
					T.name = fn;
					T.pBuildSurface = &(LM->lm_texture);
					R_ASSERT(T.pBuildSurface);
					R_ASSERT(pOGF);
					pOGF->textures.push_back(T);				 
					xr_sprintf(fn, "%s_2", LM->lm_texture.name);
					T.name = fn;
					pOGF->textures.push_back(T);
				}
			}
		}
		catch (...) 
		{
			clMsg("* ERROR: Flex2OGF, model# %d, *textures*", MODEL_ID);
		}

		// Collect faces & vertices
		F->CacheOpacity();
		bool	_tc_ = !(F->flags.bOpaque);
		try 
		{
			BuildOGFGeom(*pOGF, *faces, _tc_);
		}
		catch (...) { clMsg("* ERROR: Flex2OGF, model# %d, *faces*", MODEL_ID); }

	}
	catch (...)
	{
		clMsg("* ERROR: Flex2OGF, 1st part, model# %d", MODEL_ID);
	}

	if (! pOGF->data.vertices.size())
 		return false;

 	pOGF->Optimize();
 	pOGF->CalcBounds();
    pOGF->MakeProgressive(c_PM_MetricLimit_static);
  	pOGF->Stripify();

	return true;
};
 

#include <thread>
#include "ppl.h"
#include <wchar.h>

extern XRCORE_API BOOL			g_bEnableStatGather;
void CBuild::Flex2OGF()
{
	g_bEnableStatGather = true;

	float p_total	= 0;
	float p_cost	= 1/float(g_XSplit.size());

	validate_splits	();

	g_tree.clear	();
	g_tree.reserve	(4096);
	Status("Converting to OGF size [%d]", g_XSplit.size());

	int IDSplit = 0;
	for (auto& SPLIT : g_XSplit)
	{
		if (SPLIT == nullptr)
		{
			Msg("Problem In SPLIT: %d", IDSplit);
		}
		IDSplit++;
	}
	  
	if (g_XSplit.size() > 256)
	{
		static xrCriticalSection mtx;
		std::atomic<int> current_idx = 0;
 
		concurrency::parallel_for(size_t(0), size_t(16), [&](size_t thID)
		{
			std::wstring name = L"ThreadID : " + std::to_wstring(thID);

			SetThreadDescription(GetCurrentThread(), name.c_str());
			while (true)
			{
				u32 ID = current_idx.load();
				current_idx.fetch_add(1);
 				if (current_idx.load() >= g_XSplit.size())  break;

   				AditionalData("Processed MT OGF (%u|%u) delVert: %u", ID, g_XSplit.size());
 				 
				OGF* pOGF = xr_new<OGF>();
				auto& SPLIT = g_XSplit[ID];
				Face* Face = SPLIT->front();			// first face
 
				int VertexRemoved = 0;				 
				ConvertOgf(thID, ID, SPLIT, Face, &(materials()[Face->dwMaterial]), pOGF, this);
  
				mtx.Enter();
				g_tree.push_back(pOGF);
				mtx.Leave();
			}	
		});
 	}
	else
	{
		int Removed = 0;
		for (size_t ID = (0); ID < g_XSplit.size(); ID++)
		{
			if (ID % 512 == 0)
				clMsg("Processed (%u|%u)", ID, g_XSplit.size());

			OGF* pOGF = xr_new<OGF>();
			auto& SPLIT = g_XSplit[ID];
			Face* Face = SPLIT->front();			// first face			
			ConvertOgf(0, ID, SPLIT, Face, &(materials()[Face->dwMaterial]), pOGF, this);
			 
  			g_tree.push_back(pOGF);
		};
	}
	 
	g_XSplit.clear();
}
