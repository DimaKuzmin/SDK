#include "stdafx.h"
//#include "cl_collector.h"
#include "build.h"
#include "../xrLCLight/xrMU_Model.h"
#include "../xrLCLight/xrMU_Model_Reference.h"

#include "../xrLCLight/xrLC_GlobalData.h"
#include "../../xrcdb/xrcdb.h"
#include "../xrLCLight/xrface.h"

//.#include "communicate.h"

CDB::MODEL*	RCAST_Model	= 0;

IC bool				FaceEqual(Face& F1, Face& F2)
{
	// Test for 6 variations
	if ((F1.v[0]==F2.v[0]) && (F1.v[1]==F2.v[1]) && (F1.v[2]==F2.v[2])) return true;
	if ((F1.v[0]==F2.v[0]) && (F1.v[2]==F2.v[1]) && (F1.v[1]==F2.v[2])) return true;
	if ((F1.v[2]==F2.v[0]) && (F1.v[0]==F2.v[1]) && (F1.v[1]==F2.v[2])) return true;
	if ((F1.v[2]==F2.v[0]) && (F1.v[1]==F2.v[1]) && (F1.v[0]==F2.v[2])) return true;
	if ((F1.v[1]==F2.v[0]) && (F1.v[0]==F2.v[1]) && (F1.v[2]==F2.v[2])) return true;
	if ((F1.v[1]==F2.v[0]) && (F1.v[2]==F2.v[1]) && (F1.v[0]==F2.v[2])) return true;
	return false;
}

void SaveUVM			(LPCSTR fname, xr_vector<b_rc_face>& vm)
{
	IWriter* W			= FS.w_open(fname);
	string256 tmp;
	// vertices
	for (u32 v_idx=0; v_idx<vm.size(); v_idx++){
		b_rc_face& rcf	= vm[v_idx];
		xr_sprintf			(tmp,"f %d %d [%3.2f,%3.2f]-[%3.2f,%3.2f]-[%3.2f,%3.2f]",rcf.dwMaterial,rcf.dwMaterialGame,
						rcf.t[0].x,rcf.t[0].y, rcf.t[1].x,rcf.t[1].y, rcf.t[2].x,rcf.t[2].y);
		W->w_string		(tmp);
	}
	FS.w_close	(W);
}

size_t GetMemoryRequiredForLoadLevel(CDB::MODEL* RaycastModel, base_lighting& Lightings, xr_vector<b_BuildTexture>& Textures)
{
	size_t VertexDataSize = RaycastModel->get_verts_count() * sizeof(12);
	size_t TrisIndexSize = RaycastModel->get_tris_count() * sizeof(12);
	size_t TrisAdditionalDataSize = RaycastModel->get_tris_count() * sizeof(32);

	size_t OptixMeshDataOverhead = VertexDataSize + TrisIndexSize;

	size_t TextureMemorySize = 0;
	for (const b_BuildTexture& Texture : Textures)
	{
		size_t TextureSize = (Texture.dwHeight * Texture.dwWidth) * sizeof(u32);
		TextureSize += sizeof(24);
		TextureMemorySize += TextureSize;
	}


	size_t LightingInfoSize = (Lightings.rgb.size() + Lightings.sun.size() + Lightings.hemi.size()) * sizeof(R_Light);
	size_t TotalMemorySize = VertexDataSize + TrisIndexSize + TrisAdditionalDataSize + OptixMeshDataOverhead + TextureMemorySize + LightingInfoSize;

	clMsg(" [xrHardwareLight]: Vertex data size: %llu MB, Tris index size: %llu MB", VertexDataSize / 1024 / 1024, TrisIndexSize / 1024 / 1024);
	clMsg(" [xrHardwareLight]: Tris Additional Data: %llu MB", TrisAdditionalDataSize / 1024 / 1024);
	clMsg(" [xrHardwareLight]: OptiX overhead: %llu MB", OptixMeshDataOverhead / 1024 / 1024);
	clMsg(" [xrHardwareLight]: Overall texture memory: %llu MB", TextureMemorySize / 1024 / 1024);
	clMsg(" [xrHardwareLight]: Lighting: %llu MB", LightingInfoSize / 1024 / 1024);
	clMsg(" [xrHardwareLight]: TOTAL: %llu MB", TotalMemorySize / 1024 / 1024);

	return TotalMemorySize;
}

#include "../../xrcdb/xrcdb.h"
  
void CBuild::BuildRapid		(BOOL bSaveForOtherCompilers)
{
	float	p_total			= 0;
	float	p_cost			= 1.f/(lc_global_data()->g_faces().size());

	
	lc_global_data()->destroy_rcmodel();
	Status			("Converting faces...");
	for				(u32 fit=0; fit<lc_global_data()->g_faces().size(); fit++)	
		lc_global_data()->g_faces()[fit]->flags.bProcessed = false;

	xr_vector<Face*>			adjacent_vec;
	adjacent_vec.reserve		(6*2*3);

	CDB::CollectorPacked	CL	(scene_bb, lc_global_data()->g_vertices().size(), lc_global_data()->g_faces().size());
 
//	Status("Converting faces... (ONE CORE)");
	
	for (vecFaceIt it=lc_global_data()->g_faces().begin(); it!=lc_global_data()->g_faces().end(); ++it)
	{
		Face*	F				= (*it);
		const Shader_xrLC&	SH		= F->Shader();
		
		if (!SH.flags.bLIGHT_CastShadow)		
			continue;
		

		b_material& M = lc_global_data()->materials()[F->dwMaterial];

		/*
 		if (strstr(lc_global_data()->shaders().Get(M.shader_xrlc)->Name, "_noshadow") || strstr(lc_global_data()->shaders().Get(M.shader)->Name, "_noshadow") )
		{
			// Msg("skipped: xrlc: %s, %s", lc_global_data()->shaders().Get(M.shader_xrlc)->Name, lc_global_data()->shaders().Get(M.shader)->Name);

			F->flags.bShadowSkip = true;
			continue;
		}
		*/
 
		Progress	(float(it-lc_global_data()->g_faces().begin())/float(lc_global_data()->g_faces().size()));
				
		// Collect
		adjacent_vec.clear	();
		for (int vit=0; vit<3; ++vit)
		{
			Vertex* V = F->v[vit];
			for (u32 adj=0; adj<V->m_adjacents.size(); adj++)
			{
				adjacent_vec.push_back(V->m_adjacents[adj]);
			}
		}

		std::sort		(adjacent_vec.begin(),adjacent_vec.end());
		adjacent_vec.erase	(std::unique(adjacent_vec.begin(),adjacent_vec.end()),adjacent_vec.end());

		// Unique
		BOOL			bAlready	= FALSE;

		for (u32 ait=0; ait<adjacent_vec.size(); ++ait)
		{
			Face*	Test					= adjacent_vec[ait];
			if (Test==F)					continue;
			if (!Test->flags.bProcessed)	continue;
			if (FaceEqual(*F,*Test))
			{
				bAlready					= TRUE;
				break;
			}
		}

		//
		if (!bAlready) 
		{
			F->flags.bProcessed	= true;
			CL.add_face_D		( F->v[0]->P,F->v[1]->P,F->v[2]->P, F, F->sm_group);
		}
 
	}
	 
	// Export references
	if (bSaveForOtherCompilers)		
		Phase	("Building rcast-CFORM-mu model...");

	Status					("Models...");

	//for (u32 ref = 0; ref < mu_refs().size(); ref++)
	
	int id = 0;
	std::for_each(mu_refs().begin(), mu_refs().end(), [&](xrMU_Reference* ref ) 
	{
		id++;
		Progress(float(id / float(mu_refs().size())));
		ref->export_cform_rcast(CL);
	}
	);


	// "Building tree..
	Status					("Building search tree...");
	lc_global_data()->create_rcmodel( CL );

	extern void SaveAsSMF			(LPCSTR fname, CDB::CollectorPacked& CL);
	
	//GetMemoryRequiredForLoadLevel(lc_global_data()->RCAST_Model(), lc_global_data()->L_static(), lc_global_data()->textures());

	// save source SMF
	string_path				fn;

	bool					keep_temp_files = !!strstr(Core.Params,"-keep_temp_files");

	if (g_params().m_quality!=ebqDraft)
	{
		if (keep_temp_files)
			SaveAsSMF		(strconcat(sizeof(fn),fn,pBuild->path,"build_cform_source.smf"),CL);
	}

	// Saving for AI/DO usage
	if (bSaveForOtherCompilers)
	{
		Status					("Saving...");
		string_path				fn;
 
		xr_vector<b_rc_face>	rc_faces;
		rc_faces.resize			(CL.getTS());

		size_t rqface = (rc_faces.size() * sizeof(b_rc_face) );
		size_t tri =  (CL.getTS() * CDB::TRI::Size());
		size_t VS = (CL.getVS()*sizeof(Fvector)); 

		size_t size_Rqface	=		rqface / 1024 / 1024;
		size_t size_TRI		=		tri / 1024 / 1024;
		size_t size_VS		=		VS / 1024 / 1024;

		Status("Size: VS: %llu, TRI: %llu, RC_Faces: %llu", size_VS, size_TRI, size_Rqface );

		// Prepare faces
		for (u32 k=0; k<CL.getTS(); k++)
		{
			CDB::TRI& T			= CL.getT( k );
			base_Face* F		= (base_Face*)(T.pointer);
			b_rc_face& cf		= rc_faces[k];
			cf.dwMaterial		= F->dwMaterial;
			cf.dwMaterialGame	= F->dwMaterialGame;
			Fvector2*	cuv		= F->getTC0	();
			cf.t[0].set			(cuv[0]);
			cf.t[1].set			(cuv[1]);
			cf.t[2].set			(cuv[2]);
		}
		
		if (g_params().m_quality!=ebqDraft)
		{
			if (keep_temp_files)
				SaveUVM			(strconcat(sizeof(fn),fn,pBuild->path,"build_cform_source.uvm"),rc_faces);
		}



				
		if (size_Rqface + size_TRI + size_VS > 4096)
		{
			IWriter*		MFS_TRI		= FS.w_open(strconcat(sizeof(fn),fn,pBuild->path,"build.cform_tri"));
			IWriter*		MFS_VS		= FS.w_open(strconcat(sizeof(fn),fn,pBuild->path,"build.cform_vs"));
			IWriter*		MFS_RQ		= FS.w_open(strconcat(sizeof(fn),fn,pBuild->path,"build.cform_rq"));

			//TRI
			MFS_TRI->open_chunk			(0);

			// Header
			hdrCFORM hdr;
			hdr.version				= CFORM_CURRENT_VERSION;
			hdr.vertcount			= (u32)CL.getVS();
			hdr.facecount			= (u32)CL.getTS();
			hdr.aabb				= scene_bb;
			MFS_TRI->w					(&hdr,sizeof(hdr));

			MFS_TRI->close_chunk();

			MFS_TRI->open_chunk(1);
			for (size_t i = 0; i < CL.getTS(); i++)
	 			MFS_TRI->w(&CL.getT()[i], CDB::TRI::Size());
			MFS_TRI->close_chunk		();
 
			FS.w_close(MFS_TRI);

			// VS
			MFS_VS->open_chunk(0);
		    MFS_VS->w					(CL.getV(), (u32) CL.getVS() * sizeof(Fvector));
			MFS_VS->close_chunk();

			FS.w_close(MFS_VS);


			// RQ
			MFS_RQ->open_chunk(0);

			MFS_RQ->w					(&*rc_faces.begin(),(u32)rc_faces.size()*sizeof(b_rc_face));
			MFS_RQ->close_chunk();

			FS.w_close(MFS_RQ);
		}
		else 
		{
			IWriter*		MFS		= FS.w_open(strconcat(sizeof(fn),fn,pBuild->path,"build.cform"));
			MFS->open_chunk			(0);

			// Header
			hdrCFORM hdr;
			hdr.version				= CFORM_CURRENT_VERSION;
			hdr.vertcount			= (u32)CL.getVS();
			hdr.facecount			= (u32)CL.getTS();
			hdr.aabb				= scene_bb;
			MFS->w					(&hdr,sizeof(hdr));

			Status("Size: TRI: %llu, VS: %llu, RQ_Face: %llu", CL.getTS(), CL.getVS(), rc_faces.size() );

			// Data
			MFS->w					(CL.getV(),(u32)CL.getVS()*sizeof(Fvector));

			for (size_t i = 0; i < CL.getTS(); i++)
	 			MFS->w(&CL.getT()[i], CDB::TRI::Size());

			MFS->close_chunk		();

			MFS->open_chunk			(1);

			MFS->w					(&*rc_faces.begin(),(u32)rc_faces.size()*sizeof(b_rc_face));
			MFS->close_chunk		();

			FS.w_close(MFS);
		}


	}
}
