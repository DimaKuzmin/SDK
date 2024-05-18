#include "stdafx.h"

#include "xrMU_Model_Reference.h"
#include "xrMU_Model.h"

#include "light_point.h"
#include "fitter.h"
#include "xrface.h"
#include "xrLC_GlobalData.h"
template <typename T, typename T2>
T	simple_optimize				(xr_vector<T>& A, xr_vector<T>& B, T2& _scale, T2& _bias)
{
	T		accum;
	u32		it;


	T		scale	= _scale;
	T		bias	= _bias;
	T		error	= flt_max;
	T		elements= T(A.size());
	u32		count	= 0;
	for (;;)
	{
		count++;
		if (count>128)	{
			_scale		= (T2)scale;
			_bias		= (T2)bias;
			return error;
		}

		T	old_scale	= scale;
		T	old_bias	= bias;

		//1. scale
		u32		_ok			= 0;
		for (accum=0, it=0; it<A.size(); it++)
			if (_abs(A[it])>EPS_L)	
			{
				accum	+= (B[it]-bias)/A[it];
				_ok		+= 1;
			}
		T	s	= _ok?(accum/_ok):scale;

		//2. bias
		T	b	= bias;
		if (_abs(scale)>EPS)
		{
			for (accum=0, it=0; it<A.size(); it++)
				accum	+= B[it]-A[it]/scale;
			b	= accum	/ elements;
		}

		// mix
		T		conv	= 7;
		scale			= ((conv-1)*scale+s)/conv;
		bias			= ((conv-1)*bias +b)/conv;

		// error
		for (accum=0, it=0; it<A.size(); it++)
			accum	+= B[it] - (A[it]*scale + bias);
		T	err			= accum/elements;

		if (err<error)	
		{
			// continue?
			error	= err;
			if (error<EPS)	
			{
				_scale		= (T2)scale;
				_bias		= (T2)bias;
				return error;
			}
		}
		else
		{
			// exit
			_scale	= (T2)old_scale;
			_bias	= (T2)old_bias;
			return	error;
		}
	}
}



void	o_test (int iA, int iB, int count, base_color* A, base_color* B, float& C, float& D)
{
	xr_vector<double>	_A,_B;
	_A.resize			(count);
	_B.resize			(count);
	for (int it=0; it<count; it++)
	{
		base_color_c _a;	
		base_color_c _b;

		A[it]._get(_a);
		B[it]._get(_b);	
		float* f_a = (float*)&_a;

		float*	f_b	= (float*)&_b;

		_A[it]				= f_a[iA];
		_B[it]				= f_b[iB];
	}
	// C=1, D=0;
	simple_optimize		(_A,_B,C,D);
}

#include "BuildArgs.h"
extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

xrCriticalSection csMU;

int ID = 0;

void xrMU_Reference::calc_lighting()
{
	ID++;

	model->calc_lighting(
		color,
		xform,
		inlc_global_data()->RCAST_Model(),
		inlc_global_data()->L_static(),
		//	(inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) |
		//	(inlc_global_data()->b_nosun()? LP_dont_sun : 0) | 
		//	(inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) |
		0 | LP_DEFAULT,
		!build_args->use_MU_Lighting
	);

	R_ASSERT(color.size() == model->color.size());

	// A*C + D = B
	// build data

	// FIX MU MODELS 
	// Чинит темные Тени на деревьях

	bool tree = strstr(model->m_name.c_str(), "tree");
 
	/*
 	if (build_args->MU_ModelsRegression)
	{
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
				//B[it]		=	(__B.hemi);
				B[it] = ((float*)&__B)[i];
			}

			vfComputeLinearRegression(A, B, _s[i], _b[i]);
		}

		for (u32 index = 0; index < 5; index++)
		{
			o_test(4, index, color.size(), &model->color.front(), &color.front(), _s[index], _b[index]);
		}
   
	}
	*/


	{
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

		 // Поправка Хеми Света
		float global_hemi = 0;
		float model_hemi = 0;
		float global_sun = 0;
		float model_sun = 0;

		for (u32 it = 0; it < color.size(); it++)
		{
			base_color_c C, R;
			model->color[it]._get(R);
			color[it]._get(C);
			global_hemi += C.hemi;
			model_hemi += R.hemi;
			global_sun += C.sun;
			model_hemi += R.sun;

			if (R.hemi > C.hemi && C.hemi < 0.15f)
			{
 				// C.hemi = R.hemi - 0.05f;
				// color[it]._set(C);
				// C.rgb.set(255, 0, 0);
				// color[it]._set(C);
			}

		}
		 
		global_hemi = global_hemi / color.size();
		model_hemi = model_hemi / color.size();
 
		global_sun = global_sun / color.size();
		model_sun = model_sun / color.size();

		 
		// if (_s[3] < 0 || _s[4] < 0 || _b[3] < 0 || _b[4] < 0)
		{
			csMU.Enter();

			clMsg("MU: Global Hemi: %f, Model : %f", global_hemi, model_hemi);
			clMsg("MU: Global Sun: %f, Model : %f", global_sun, model_sun);

			clMsg("MU: Name: %s", model->m_name.c_str());
			clMsg("MU: ColorScale: %f, %f, %f, %f, %f", _s[0], _s[1], _s[2], _s[3], _s[4] );
			clMsg("MU: ColorBias: %f, %f, %f, %f, %f", _s[0], _b[1], _b[2], _b[3], _b[4]);
			clMsg("MU: Pos: %f, %f, %f", VPUSH(this->xform.c));
			csMU.Leave();

			//if (_s[3] < 0)
			//	_s[3] = 0;
			//if (_s[4] < 0)
			//	_s[4] = 0;
		}
 	
	}



}
