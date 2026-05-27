#include "stdafx.h"
#include "xrMU_Model.h"
#include "xrMU_Model_Reference.h"

#include "../../xrcdb/xrcdb.h"
#include "../Public/shader_xrlc.h"
 
void xrMU_Model::export_cform_rcast_new(xr_vector<FaceDataEmbree>& faces, Fmatrix& xform)
{
	v_faces			adjacent;	adjacent.reserve(6 * 2 * 3);

	// Добовляю все что есть потом отсортирую !
	for (auto F : m_faces)
	{
 		const Shader_xrLC& SH = F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)	continue;
 
		Fvector					P[3];
		xform.transform_tiny(P[0], F->v[0]->P);
		xform.transform_tiny(P[1], F->v[1]->P);
		xform.transform_tiny(P[2], F->v[2]->P);
			
		FaceDataEmbree data;
		data.v1 = P[0];
		data.v2 = P[1];
		data.v3 = P[2];
		data.ptr = F;
		faces.push_back(data);
	}
}
  
void xrMU_Model::export_cform_rcast	(CDB::CollectorPacked& CL, Fmatrix& xform)
{
	v_faces			adjacent;
	adjacent.reserve(6*2*3);

	for (auto F : m_faces)
	{
 		const Shader_xrLC&	SH		= F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)	 continue;
 
		Fvector					P[3];
		xform.transform_tiny	(P[0],F->v[0]->P);
		xform.transform_tiny	(P[1],F->v[1]->P);
		xform.transform_tiny	(P[2],F->v[2]->P);
		CL.add_face_D			(P[0],P[1],P[2], F, F->sm_group );//
	}
}

