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



	if (build_args->MU_ModelsRegression)
	{
		/*
		csMU.Enter();

		string_path file;
		string128 tmp32;
		sprintf(tmp32, "MUMODELS\\%s\\%d.txt", model->m_name.c_str(), ID);
		FS.update_path(file, "$fs_root$", tmp32);

		IWriter* w = FS.w_open(file);

		FPU::m64r			();

		struct PreRegresion
		{
			int ID;
			float hemi, hemi_r;
			float sun_original;
			float sun_calculated;

			float _s[5];
			float _b[5];
		};

		xr_vector<PreRegresion> colors_pre;
		colors_pre.resize(color.size());


		xr_vector<double>	A;	A.resize(color.size());
		xr_vector<double>	B;	B.resize(color.size());
		float* _s = (float*)&c_scale;
		float* _b = (float*)&c_bias;

		for (u32 i=0; i<5; i++)
		{
			for (u32 it=0; it<color.size(); it++)
			{
				base_color_c		__A;	model->color	[it]._get(__A);
				base_color_c		__B;	color			[it]._get(__B);
				A[it]		= 	(__A.hemi);
				//B[it]		=	(__B.hemi);
				B[it]		=	((float*)&__B)[i];
			}

			vfComputeLinearRegression(A,B,_s[i],_b[i]);
		}



		if (tree)
		for (u32 it = 0; it < color.size(); it++)
		{
			base_color_c C;
			color[it]._get(C);
			base_color_c r;
			model->color[it]._get(r);

			//string128 tmp_msg;
			//sprintf(tmp_msg, "[%d] PRE OTEST Hemi: %f cmp %f, \\n", it, C.hemi, r.hemi);
			//w->w_string(tmp_msg);
			//sprintf(tmp_msg, "		REGRESSION A: %f, %f, %f, %f, %f \\n", _s[0], _s[1], _s[2], _s[3], _s[4]);
			//w->w_string(tmp_msg);
			//sprintf(tmp_msg, "		REGRESSION B: %f, %f, %f, %f, %f \\n", _b[0], _b[1], _b[2], _b[3], _b[4]);
			//w->w_string(tmp_msg);

			PreRegresion data;
			data._s[0] = _s[0];
			data._s[1] = _s[1];
			data._s[2] = _s[2];
			data._s[3] = _s[3];
			data._s[4] = _s[4];

			data._b[0] = _b[0];
			data._b[1] = _b[1];
			data._b[2] = _b[2];
			data._b[3] = _b[3];
			data._b[4] = _b[4];

			data.hemi = C.hemi;
			data.hemi_r = r.hemi;
			data.sun_calculated = C.sun;
			data.sun_original = r.sun;


			colors_pre[it] = data;

		}

		for (u32 index = 0; index < 5; index++)
		{
			o_test(4, index, color.size(), &model->color.front(), &color.front(), _s[index], _b[index]);
		}

		if (tree)
		for (u32 it = 0; it < color.size(); it++)
		{
			base_color_c C;
			color[it]._get(C);
			base_color_c r;
			model->color[it]._get(r);

			auto predata = colors_pre[it];

			string128 tmp_msg;
			sprintf(tmp_msg, "[%d] PRE Hemi: %f cmp %f, Sun: %f, Orig: %f\\n", it, predata.hemi, predata.hemi_r, predata.sun_original, predata.sun_calculated);
			w->w_string(tmp_msg);
			sprintf(tmp_msg, "		REGRESSION A: %f, %f, %f, %f, %f \\n", predata._s[0], predata._s[1], predata._s[2], predata._s[3], predata._s[4]);
			w->w_string(tmp_msg);
			sprintf(tmp_msg, "		REGRESSION B: %f, %f, %f, %f, %f \\n", predata._b[0], predata._b[1], predata._b[2], predata._b[3], predata._b[4]);
			w->w_string(tmp_msg);


			sprintf(tmp_msg, "[%d] Hemi: %f cmp %f, \\n", it, C.hemi, r.hemi);
			w->w_string(tmp_msg);
			sprintf(tmp_msg, "		REGRESSION A: %f, %f, %f, %f, %f \\n", _s[0], _s[1], _s[2], _s[3], _s[4]);
			w->w_string(tmp_msg);
			sprintf(tmp_msg, "		REGRESSION B: %f, %f, %f, %f, %f \\n", _b[0], _b[1], _b[2], _b[3], _b[4]);
			w->w_string(tmp_msg);
		}


		FS.w_close(w);


		csMU.Leave();
		*/
	}
	else
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

		 
		for (u32 it = 0; it < color.size(); it++)
		{
			base_color_c C, R;
			model->color[it]._get(R);
			color[it]._get(C);

			if (R.hemi > C.hemi && C.hemi < 0.15f)
			{
				C.hemi = R.hemi - 0.05f;
				color[it]._set(C);
			}
		}
		 
		if (_s[3] < 0 || _s[4] < 0 || _b[3] < 0 || _b[4] < 0)
		{
			csMU.Enter();
			clMsg("MU: Name: %s", model->m_name.c_str());
			clMsg("MU: ColorScale: %f, %f, %f, %f, %f", _s[0], _s[1], _s[2], _s[3], _s[4]);
			clMsg("MU: ColorBias: %f, %f, %f, %f, %f", _s[0], _b[1], _b[2], _b[3], _b[4]);
			clMsg("MU: Pos: %f, %f, %f", VPUSH(this->xform.c));
			csMU.Leave();
		}
	
	}


}
