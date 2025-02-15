// Sector.cpp: implementation of the CSector class.
//
//////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "Build.h"
#include "Sector.h"
#include "OGF_Face.h"
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSector::CSector(u32 ID)
{
	SelfID = ID;
	TreeRoot = 0;
}

CSector::~CSector()
{

}

IC BOOL	ValidateMerge(Fbox& bb_base, Fbox& bb, float& volume, float SLimit)
{
	// Size
	Fbox	merge;
	merge.merge(bb_base, bb);
	
	Fvector sz;
	merge.getsize(sz);
	sz.add(EPS_L);

	if (sz.x > SLimit)		return FALSE;	// Don't exceed limits (4/3 GEOM)
	if (sz.y > SLimit)		return FALSE;
	if (sz.z > SLimit)		return FALSE;

	// Volume
	volume = merge.getvolume();

	// OK
	return TRUE;
}

void CSector::BuildHierrarhy()
{
	Fvector		scene_size;
	float		delimiter;
	BOOL		bAnyNode = FALSE;

	// calc scene BB
	Fbox& scene_bb = pBuild->scene_bb;
	scene_bb.invalidate();
	for (int I = 0; I < s32(g_tree.size()); I++)
		scene_bb.merge(g_tree[I]->bbox);
	scene_bb.grow(EPS_L);

	// 
	scene_bb.getsize(scene_size);
	delimiter = _max(scene_size.x, _max(scene_size.y, scene_size.z));
	delimiter *= 2;

	clMsg("Scene Size MIN{%f, %f, %f} MAX{%f, %f, %f}", VPUSH(scene_size), VPUSH(scene_size));
	clMsg("Scene Delimiter: {%f}", delimiter);

	int		iLevel = 2;
	float	SizeLimit = c_SS_maxsize / 4.f;
	if (SizeLimit < 4.f)			SizeLimit = 4.f;
	if (delimiter <= SizeLimit)	delimiter *= 2;		// just very small level

	struct OGF_DATA
	{
		u32 ID;
		OGF_Base* OGF;
	};

	for (; SizeLimit <= delimiter; SizeLimit *= 2)
	{
		int iSize = (int)g_tree.size();


		xr_vector<OGF_DATA> OGFS_SECTOR;

		int IDs = 0;
		for (auto OGF : g_tree)
		{
			if (OGF->Sector == SelfID && !OGF->bConnected)
			{
				OGF_DATA data;
				data.ID = IDs;
				data.OGF = OGF;
				OGFS_SECTOR.push_back(data);
			}
			IDs++;
		}

		string64 tmp;
		sprintf_s(tmp, "Sectors Processing (%llu) | (%llu) no connected (%d)", (size_t)SizeLimit, (size_t)iSize, OGFS_SECTOR.size()); // 
		Status(tmp);

		auto find_best = [&](OGF_Node* pNode)
			{
				while (true)
				{
					// Find best object to connect with
					// se7kills MT STYLE FIND
					int best_id = -1;
					float best_volume = flt_max;

					for (auto OGF : OGFS_SECTOR)
					{
						OGF_Base* candidate = g_tree[OGF.ID];
						if (candidate->bConnected)			continue;
						if (candidate->Sector != SelfID)	continue;
 						float V;
						if (ValidateMerge(pNode->bbox, candidate->bbox, V, SizeLimit))
						{
							if (V < best_volume)
							{
								best_volume = V;
								best_id = OGF.ID;
							}
						}
					}

					// Analyze
					if (best_id < 0) break;
					pNode->AddChield(best_id);
				}
			};
		 
		CTimer t, TPhase; 
		t.Start();
		u32 Finding = 0;
		for (auto O : OGFS_SECTOR)
		{
			int I = O.ID;

			if (g_tree[I]->bConnected)
				continue;
			if (g_tree[I]->Sector != SelfID)
				continue;

 			OGF_Node* pNode = new OGF_Node(iLevel, u16(SelfID));
			pNode->AddChield(I);
	 
			// Find best object to connect with
			TPhase.Start();
			find_best(pNode);
			Finding += TPhase.GetElapsed_ms();

			// Chields
 			if (pNode->chields.size() > 1)
			{
 				pNode->CalcBounds();
 				g_tree.push_back(pNode);
				bAnyNode = TRUE;
			}
			else
			{
				g_tree[I]->bConnected = false;
				xr_delete(pNode);
			}
 
		}

		if (g_tree.size() > 10000)
		{
			extern u64 MSSphereV1; extern u64 MSSphereV2; extern u64 MSSphereV3; extern u64 MSVALIDATION;
			clMsg(
					"Connections time[%u]: SizeLimit[%u]"
					"(CalcBounds) Finding Spheres FIND BEST(%u) ms : CALC BOUNDS: V1: %u ms, V2: %u ms, V3: %u ms, Validation: %u ms", 
					t.GetElapsed_ms(),  SizeLimit, Finding,
					MSSphereV1, MSSphereV2, MSSphereV3, MSVALIDATION
			);
		}
	

		OGFS_SECTOR.clear();

		if (iSize != (int)g_tree.size())
			iLevel++;
	}


	TreeRoot = 0;


	int TreeRootID = 0;

	if (bAnyNode)
	{
		TreeRoot = g_tree.back();
	}
	else
	{
		for (u32 I = 0; I < g_tree.size(); I++)
		{
			if (g_tree[I]->bConnected)
				continue;
			if (g_tree[I]->Sector != SelfID)
				continue;

			R_ASSERT(0 == TreeRoot);
			TreeRoot = g_tree[I];
			TreeRootID = I;
		}
	}


	int IDxx = 0;
	for (auto O : g_tree)
	{
 		if (!O->bConnected && O->Sector == SelfID && TreeRoot != O)
			clMsg("Sector[%u] Geom Tree[%u] !! is No Connection, OBJSector[%u]", SelfID, IDxx, O->Sector);
 
		IDxx++;
	}

	if (TreeRoot)
	{
		OGF_Node* node = dynamic_cast<OGF_Node*>(TreeRoot);


		clMsg("[Sector] Tree Root [%d] ch[%d]: FBOX[%.2f,%.2f,%f][%.2f,%.2f,%.2f]",
			TreeRootID,
			node ? node->chields.size() : -1,
			VPUSH(TreeRoot->bbox.min),
			VPUSH(TreeRoot->bbox.max));
	}

	if (0 == TreeRoot)
	{
		clMsg("Can't build hierrarhy for sector #%d", SelfID);
	}
}

void CSector::Validate()
{
	std::sort(Portals.begin(), Portals.end());
	R_ASSERT(std::unique(Portals.begin(), Portals.end()) == Portals.end());
	R_ASSERT(TreeRoot);
	R_ASSERT(TreeRoot->Sector == SelfID);
}

void CSector::Save(IWriter& fs)
{
	// Root
	xr_vector<OGF_Base*>::iterator F = std::find(g_tree.begin(), g_tree.end(), TreeRoot);
	R_ASSERT(F != g_tree.end());
	u32 ID = u32(F - g_tree.begin());
	fs.w_chunk(fsP_Root, &ID, sizeof(u32));

	// Portals
	fs.w_chunk(fsP_Portals, &*Portals.begin(), (u32)Portals.size() * sizeof(u16));
}
