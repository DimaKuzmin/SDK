#include "stdafx.h"
#include "build.h"
 
/*
void SaveUVM(LPCSTR fname, xr_vector<b_rc_face>& vm)
{
	IWriter* W = FS.w_open(fname);
	string256 tmp;
	// vertices
	for (u32 v_idx = 0; v_idx < vm.size(); v_idx++) {
		b_rc_face& rcf = vm[v_idx];
		xr_sprintf(tmp, "f %d %d [%3.2f,%3.2f]-[%3.2f,%3.2f]-[%3.2f,%3.2f]", rcf.dwMaterial, rcf.dwMaterialGame,
			rcf.t[0].x, rcf.t[0].y, rcf.t[1].x, rcf.t[1].y, rcf.t[2].x, rcf.t[2].y);
		W->w_string(tmp);
	}
	FS.w_close(W);
}

void SaveAsSMF(LPCSTR fname, CDB::CollectorPacked& CL) -> OBJ
{
	IWriter* W = FS.w_open(fname);
	string256 tmp;
	// vertices
	for (u32 v_idx = 0; v_idx < CL.getVS(); v_idx++) {
		Fvector* v = CL.getV() + v_idx;
		xr_sprintf(tmp, "v %f %f %f", v->x, v->y, -v->z);
		W->w_string(tmp);
	}
	// transfer faces
	for (u32 f_idx = 0; f_idx < CL.getTS(); f_idx++) {
		CDB::TRI& t = CL.getT(f_idx);
		xr_sprintf(tmp, "f %d %d %d", t.verts[0] + 1, t.verts[2] + 1, t.verts[1] + 1);
		W->w_string(tmp);
	}
	FS.w_close(W);
}
*/
 