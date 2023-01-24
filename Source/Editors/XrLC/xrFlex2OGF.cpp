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
	for (splitIt it=g_XSplit.begin(); it!=g_XSplit.end(); it++)
	{
		u32 MODEL_ID		= u32(it-g_XSplit.begin())	;
		if ((*it)->size() > c_SS_HighVertLimit*2)		{
			clMsg	("! ERROR: subdiv #%d has more than %d faces (%d)",MODEL_ID,2*c_SS_HighVertLimit,(*it)->size());
		}
	};
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

xr_vector<int> thread_list;
xrCriticalSection csOGF;
 

void ThreadOgf(bool use_mt, u32 MODEL_ID,  vecFace* faces , Face* F, b_material* M, OGF* pOGF, CBuild* build)
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
				if (LM) {
					string_path	fn;
					xr_sprintf(fn, "%s_1", LM->lm_texture.name);
					T.name = fn;
					T.pBuildSurface = &(LM->lm_texture);
					R_ASSERT(T.pBuildSurface);
					R_ASSERT(pOGF);
					pOGF->textures.push_back(T);					//.
					xr_sprintf(fn, "%s_2", LM->lm_texture.name);
					T.name = fn;
					pOGF->textures.push_back(T);
				}
			}
		}
		catch (...) { clMsg("* ERROR: Flex2OGF, model# %d, *textures*", MODEL_ID); }

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
	  
	try
	{
		pOGF->use_mt_progresive = use_mt;

		csOGF.Enter();
 		clMsg("%3d: opt : v(%d)-f(%d)", MODEL_ID, pOGF->data.vertices.size(), pOGF->data.faces.size());
		pOGF->Optimize();
 		
		clMsg("%3d: cb  : v(%d)-f(%d)", MODEL_ID, pOGF->data.vertices.size(), pOGF->data.faces.size());
		pOGF->CalcBounds();
 		csOGF.Leave();
		
 		clMsg("%3d: prog: v(%d)-f(%d)", MODEL_ID, pOGF->data.vertices.size(), pOGF->data.faces.size());
		pOGF->MakeProgressive(MODEL_ID, c_PM_MetricLimit_static);
 
		csOGF.Enter();
		clMsg("%3d: strp: v(%d)-f(%d)", MODEL_ID, pOGF->data.vertices.size(), pOGF->data.faces.size());
		pOGF->Stripify();
		csOGF.Leave();
		 
	}
	catch (...) 
	{
		clMsg("* ERROR: Flex2OGF, 2nd part, model# %d", MODEL_ID);
	}
 
};

void MainThreadOGF(CBuild* build, int thID, bool use_mt_progresive)
{
	for (;;)
	{	 
		csOGF.Enter();

		if (thread_list.empty())
		{
			csOGF.Leave();
			break;
		}

		int id = thread_list.back();
		thread_list.pop_back();
		csOGF.Leave();	

		OGF* pOGF = xr_new<OGF>();
		Face* F = g_XSplit[id]->front();			// first face
		b_material* M = &(build->materials()[F->dwMaterial]);	// and it's material

		ThreadOgf(use_mt_progresive, id, g_XSplit[id], F, M, pOGF, build);
		
		g_tree.push_back(pOGF);
 	}
};

#include <thread>
  
int THREADS_COUNT();
#define MAX_THREADS THREADS_COUNT()


//#define USE_MT 

void CBuild::Flex2OGF()
{
	float p_total	= 0;
	float p_cost	= 1/float(g_XSplit.size());

	validate_splits	();

	g_tree.clear	();
	g_tree.reserve	(4096);
	Status("Converting to OGF size [%d]", g_XSplit.size());

	//for (splitIt it = g_XSplit.begin(); it != g_XSplit.end(); it++)
	for (int i = 0; i < g_XSplit.size(); i++)
		thread_list.push_back(i);

#ifdef USE_MT
	std::thread* th = new std::thread[8];

	for (int i = 0; i < MAX_THREADS; i++)
		th[i] = std::thread(MainThreadOGF, this, i, true);

	for (int i = 0; i < MAX_THREADS; i++)
		th[i].join();
#else
	MainThreadOGF(this, 0, false);
#endif

	g_XSplit.clear_and_free();
}
