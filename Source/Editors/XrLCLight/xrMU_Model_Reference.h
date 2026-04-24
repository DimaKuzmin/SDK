#ifndef XRMUMODEL_REFERENCE_H
#define XRMUMODEL_REFERENCE_H

#include "base_color.h"
#include "embree_raytracing/EmbreeRayTrace.h"

#include <ppl.h> 
#include <concurrent_unordered_map.h>

struct FaceDataIntel;
class xrMU_Model;
namespace CDB { class CollectorPacked; }
 

class XRLC_LIGHT_API xrMU_Reference
{
public:
	xrMU_Model*				model;
    Fmatrix					xform;
    Flags32					flags;
	u16						sector;

	xr_vector<base_color>	color;

	base_color_c			c_scale;
	base_color_c			c_bias;
public:
							xrMU_Reference		(): model(0), sector(u16(-1)), flags(Flags32().assign(0)), xform(Fidentity){}

	void					Load				( IReader& fs, xr_vector<xrMU_Model*>& mu_models );
	void					calc_lighting		();

	void					export_cform_game	(CDB::CollectorPacked& CL);
	void					export_cform_rcast	(CDB::CollectorPacked& CL);

	void					export_cform_rcast_new  (xr_vector<FaceDataIntel>& faces);
 
	// Cuda Code: 
	concurrency::concurrent_unordered_map<size_t, base_color_c> colors_cuda;

	void calc_lighting_cuda_1();
	void calc_lighting_cuda_2();
	void calc_lighting_cuda_3();


};
#endif