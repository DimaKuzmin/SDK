#include "stdafx.h"

#include "xrMU_Model_Reference.h"
#include "xrMU_Model.h"

#include "light_point.h"
#include "fitter.h"
#include "xrface.h"
#include "xrLC_GlobalData.h"
#include "xrMu_Resampling.h"

void o_test(int iA, int iB, int count, base_color* A, base_color* B, float& C, float& D)
{
	xr_vector<double>	_A, _B;
	_A.resize(count);
	_B.resize(count);
	for (int it = 0; it < count; it++)
	{
		base_color_c _a;
		base_color_c _b;

		A[it]._get(_a);
		B[it]._get(_b);
		float* f_a = (float*)&_a;

		float* f_b = (float*)&_b;

		_A[it] = f_a[iA];
		_B[it] = f_b[iB];
	}
	// C=1, D=0;
	simple_optimize(_A, _B, C, D);
}

void xrMU_Reference::calc_lighting()
{
	model->calc_lighting(
		color,
		xform,
		inlc_global_data()->RCAST_Model(),
		inlc_global_data()->L_static(),
		LP_DEFAULT
	);

	R_ASSERT(color.size() == model->color.size());

	// A*C + D = B
	// build data
 	xr_vector<double>	A;	A.resize(color.size());
	xr_vector<double>	B;	B.resize(color.size());
	float* _s = (float*)&c_scale;
	float* _b = (float*)&c_bias;

	for (u32 i = 0; i < 5; i++)
	{
		for (u32 it = 0; it < color.size(); it++)
		{
			base_color_c		__A;
			model->color[it]._get(__A);
			base_color_c		__B;
			color[it]._get(__B);
			A[it] = (__A.hemi);
 			B[it] = ((float*)&__B)[i];
		}

		vfComputeLinearRegression(A, B, _s[i], _b[i]);
	}

	for (u32 index = 0; index < 5; index++)
	{
		o_test(4, index, color.size(), &model->color.front(), &color.front(), _s[index], _b[index]);
	}
}