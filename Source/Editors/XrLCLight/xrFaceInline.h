
IC	BOOL	DataVertex::similar	(Vertex &V, float eps)
{
	return P.similar(V.P,eps);	
}


#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>


IC  BOOL	DataVertex::similar_avx		( Tvertex<DataVertex> &V, float eps )
{
	__m128 V_1 = _mm_load_ps((float*) &P);
	__m128 V_2 = _mm_load_ps((float*) &V.P);

	__m128 res = _mm_sub_ps(V_1, V_2);
	 
	return res.m128_f32[0] < eps && res.m128_f32[1] < eps && res.m128_f32[2] < eps;
}








