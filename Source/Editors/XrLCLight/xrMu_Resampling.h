#pragma once
template <typename T, typename T2>
T	simple_optimize(xr_vector<T>& A, xr_vector<T>& B, T2& _scale, T2& _bias)
{
	T		accum;
	u32		it;


	T		scale = _scale;
	T		bias = _bias;
	T		error = flt_max;
	T		elements = T(A.size());
	u32		count = 0;
	for (;;)
	{
		count++;
		if (count > 128) {
			_scale = (T2)scale;
			_bias = (T2)bias;
			return error;
		}

		T	old_scale = scale;
		T	old_bias = bias;

		//1. scale
		u32		_ok = 0;
		for (accum = 0, it = 0; it < A.size(); it++)
			if (_abs(A[it]) > EPS_L)
			{
				accum += (B[it] - bias) / A[it];
				_ok += 1;
			}
		T	s = _ok ? (accum / _ok) : scale;

		//2. bias
		T	b = bias;
		if (_abs(scale) > EPS)
		{
			for (accum = 0, it = 0; it < A.size(); it++)
				accum += B[it] - A[it] / scale;
			b = accum / elements;
		}

		// mix
		T		conv = 7;
		scale = ((conv - 1) * scale + s) / conv;
		bias = ((conv - 1) * bias + b) / conv;

		// error
		for (accum = 0, it = 0; it < A.size(); it++)
			accum += B[it] - (A[it] * scale + bias);
		T	err = accum / elements;

		if (err < error)
		{
			// continue?
			error = err;
			if (error < EPS)
			{
				_scale = (T2)scale;
				_bias = (T2)bias;
				return error;
			}
		}
		else
		{
			// exit
			_scale = (T2)old_scale;
			_bias = (T2)old_bias;
			return	error;
		}
	}
}

void o_test(int iA, int iB, int count, base_color* A, base_color* B, float& C, float& D);
