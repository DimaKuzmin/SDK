// Sector.cpp: implementation of the CSector class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "build.h"
#include "Sector.h"
#include "OGF_Face.h"
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSector::CSector(u32 ID)
{
	SelfID = ID;
	TreeRoot=0;
}

CSector::~CSector()
{

}

IC	Fbox	box_merge_avx		(Fbox b1, Fbox b2);
IC	Fvector	box_getsize_avx		(Fbox box);
IC	float	box_getvolume_avx	(Fbox box);
IC  void	box_modify_avx		(Fbox& box, Fvector& p);  

extern bool use_avx;

IC BOOL	ValidateMerge	(Fbox& bb_base, Fbox& bb, float& volume, float SLimit)
{
	// Size
	Fbox	merge = box_merge_avx		(bb_base,bb);  	
	
	Fvector sz;		
	merge.getsize	(sz);
	sz.add	(EPS_L);

	if (sz.x>SLimit)		return FALSE;	// Don't exceed limits (4/3 GEOM)
	if (sz.y>SLimit)		return FALSE;
	if (sz.z>SLimit)		return FALSE;

	// Volume
	//volume		= merge.getvolume	();
	
	if (use_avx)
		volume = box_getvolume_avx(merge);
	else 
		volume =  merge.getvolume	();

	// OK
	return TRUE;
}
 
#include <execution>

void CSector::BuildHierrarhy	()
{
	Fvector		scene_size;
	float		delimiter;
	BOOL		bAnyNode		= FALSE;

	// calc scene BB
	Fbox&		scene_bb		= pBuild->scene_bb;
	scene_bb.invalidate			();
	for (int I=0; I<s32(g_tree.size()); I++)
		scene_bb.merge			(g_tree[I]->bbox);
	scene_bb.grow	(EPS_L);

	// 
	scene_bb.getsize(scene_size);
	delimiter		=	_max(scene_size.x,_max(scene_size.y,scene_size.z));
	delimiter		*=	2;

	int		iLevel					= 2;
	float	SizeLimit				= c_SS_maxsize/4.f;
	if		(SizeLimit<4.f)			SizeLimit=4.f;
	if		(delimiter<=SizeLimit)	delimiter*=2;		// just very small level
 
	int ids_try = 0;

	for (; SizeLimit<=delimiter; SizeLimit*=2)
	{
		ids_try++;

		int iSize			= g_tree.size();

		for (int I=0; I<iSize; I++)
		{
			if (g_tree[I]->bConnected)		 continue;
			if (g_tree[I]->Sector != SelfID) continue;

			OGF_Node* pNode					= xr_new<OGF_Node>(iLevel,u16(SelfID));
			pNode->AddChield				(I);

			// Find best object to connect with
			
		 
			for (;;) 
			{
				// Find best object to connect with
				/*
				int		best_id		= -1;
				float	best_volume	= flt_max;
			   	  
				for (int J=0; J<iSize; J++)
				{
					OGF_Base* candidate = g_tree[J];
					if ( candidate->bConnected || candidate->Sector != SelfID)
						continue ;

					float V;
					if (ValidateMerge(pNode->bbox,candidate->bbox,V,SizeLimit))
					{
						if (V<best_volume)	
						{
							best_volume		= V;
							best_id			= J;
							break;
						}
					}
				};
				
				// Analyze
				if (best_id<0)		
					break;
				*/

				std::atomic<float>	best_volume	= flt_max;
				// 
				auto it = std::find_if(std::execution::par, g_tree.begin(), g_tree.end(), [&](OGF_Base* candidate)
				{
					if ( candidate->bConnected || candidate->Sector != SelfID)
						return false ;

					float V;
					if (ValidateMerge(pNode->bbox, candidate->bbox,V,SizeLimit))
					{
						if (V<best_volume)	
						{
							best_volume		= V;
 							return true;
						}
					} 

					return false;
				});

				if (it == g_tree.end())
					break;

				pNode->AddChield	(std::distance(g_tree.begin(),it));
				//pNode->AddChield	(best_id);	 
			}
		 
			/*
			std::thread* th[8];

			for (auto i = 0; i < 8; i++)
				th[i] = new std::thread(ThreadWork, pNode, SelfID, SizeLimit);

			for (auto i = 0; i < 8; i++)
				th[i]->join();
			*/

			StatusNoMSG("ID_MERGE_TRY[%d], Sector: %d/%d geometry_tree: %d, SizeLimit: %f, Delimiter: %f", ids_try, I, iSize, g_tree.size(), SizeLimit, delimiter);

			if (pNode->chields.size()>1)	
			{
				pNode->CalcBounds		();
				g_tree.push_back		(pNode);
				bAnyNode				= TRUE;
			}
			else 
			{
				g_tree[I]->bConnected	= false;
				xr_delete				(pNode);
			}
		}
		
		if (iSize != (int)g_tree.size()) 
			iLevel++;
	}
	
	TreeRoot = 0;
	
	if (bAnyNode) 
		TreeRoot = g_tree.back();
	else {
		for (u32 I=0; I<g_tree.size(); I++)
		{
			if (g_tree[I]->bConnected)		 continue;
			if (g_tree[I]->Sector != SelfID) continue;
			R_ASSERT	(0==TreeRoot);
			TreeRoot	= g_tree[I];
		}
	}
	if (0==TreeRoot) 
	{
		clMsg("Can't build hierrarhy for sector #%d",SelfID);
	}
}

void CSector::Validate()
{
	std::sort(Portals.begin(),Portals.end());
	R_ASSERT(std::unique(Portals.begin(),Portals.end())==Portals.end());
	R_ASSERT(TreeRoot);
	R_ASSERT(TreeRoot->Sector == SelfID);
}

void CSector::Save(IWriter &fs)
{
	// Root
	xr_vector<OGF_Base *>::iterator F = std::find(g_tree.begin(),g_tree.end(),TreeRoot);
	R_ASSERT(F!=g_tree.end());
	u32 ID = u32(F-g_tree.begin());
	fs.w_chunk(fsP_Root,&ID,sizeof(u32));

	// Portals
	fs.w_chunk(fsP_Portals,&*Portals.begin(),Portals.size()*sizeof(u16));
}
