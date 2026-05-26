#include "stdafx.h"
#include "build.h"
#include "ogf_face.h"
#pragma warning(disable:4995)
#include "../freemagic/MgcCont3DMinSphere.h"


BOOL	f_valid		(float f)
{
	return _finite(f) && !_isnan(f);
}

BOOL				SphereValid	(xr_vector<Fvector>& geom, Fsphere& test)
{
	if (!f_valid(test.P.x) || !f_valid(test.R))
	{
		clMsg	("*** Attention ***: invalid sphere: %f,%f,%f - %f",test.P.x,test.P.y,test.P.z,test.R);
	}

	Fsphere	S	=	test;
	S.R			+=	EPS_L;
	for (xr_vector<Fvector>::iterator	I = geom.begin(); I!=geom.end(); I++)
		if (!S.contains(*I))	return FALSE;
	return TRUE;
}

void	Sphere_Compute(Fbox& bbox, xr_vector<Fvector>& V, Fsphere& Sphere)
{
 	bbox.invalidate();
	for (auto I = V.begin(); I != V.end(); I++)
		bbox.modify(*I);

	bbox.grow(EPS_L);
	bbox.getsphere(Sphere.P, Sphere.R);
	Sphere.R = -1;

	for (auto I = V.begin(); I != V.end(); I++) 
	{
		float d = Sphere.P.distance_to_sqr(*I);
		if (d > Sphere.R)
			Sphere.R = d;
	}

	Sphere.R = _sqrt(_abs(Sphere.R));
}

void SphereComputeMagic(xr_vector<Fvector>& V, Fsphere& S3)
{
	Mgc::Sphere _S3 = Mgc::MinSphere((u32)V.size(), (const Mgc::Vector3*)&*V.begin());
  	S3.P.set(_S3.Center().x, _S3.Center().y, _S3.Center().z);
	S3.R = _S3.Radius();
}

extern u64 MSSphereV1 = 0;
extern u64 MSSphereV2 = 0;
extern u64 MSSphereV3 = 0;
extern u64 MSVALIDATION = 0;

Fsphere CalculateSphere(xr_vector<Fvector>& V, Fbox& bbox)
{
	Fsphere S2;
	bbox.invalidate();
	for (auto& I : V)
		bbox.modify(I);

	bbox.grow(EPS_L);
	bbox.getsphere(S2.P, S2.R);

	S2.R = -1;
	for (auto& I : V)
	{
		float d = S2.P.distance_to_sqr(I);
		if (d > S2.R)
			S2.R = d;
	}

	S2.R = _sqrt(_abs(S2.R));
	return S2;
}

Fsphere CalculateMagic(xr_vector<Fvector>& V)
{
	Mgc::Sphere _S3 = Mgc::MinSphere((u32)V.size(), (const Mgc::Vector3*)&*V.begin());

	Fsphere	S3;
	S3.P.set(_S3.Center().x, _S3.Center().y, _S3.Center().z);
	S3.R = _S3.Radius();
	return S3;
}



void				OGF_Base::CalcBounds()
{
	// get geometry
	xr_vector<Fvector>		V;
	xr_vector<Fvector>::iterator	I;
	V.clear();
	V.reserve(4096);
	GetGeometry(V);
 	R_ASSERT(V.size() >= 3);

	// 1: calc first variation
	Fsphere	S1;
	Fsphere_compute(S1, &*V.begin(), (u32)V.size());
	BOOL B1 = SphereValid(V, S1);

	// 2: calc ordinary algorithm (2nd)
	Fsphere	S2;
	bbox.invalidate();
	for (I = V.begin(); I != V.end(); I++)
		bbox.modify(*I);
	bbox.grow(EPS_L);
	bbox.getsphere(S2.P, S2.R);
	S2.R = -1;
	for (I = V.begin(); I != V.end(); I++) {
		float d = S2.P.distance_to_sqr(*I);
		if (d > S2.R) 
			S2.R = d;
	}
	S2.R = _sqrt(_abs(S2.R));
	BOOL B2 = SphereValid(V, S2);

	// 3: calc magic-fm
	Mgc::Sphere _S3 = Mgc::MinSphere((u32)V.size(), (const Mgc::Vector3*)&*V.begin());
	Fsphere	S3;
	S3.P.set(_S3.Center().x, _S3.Center().y, _S3.Center().z);
	S3.R = _S3.Radius();
	BOOL B3 = SphereValid(V, S3);

	// select best one
	if (B1 && (S1.R < S2.R))
	{
		// miniball or FM
		if (B3 && (S3.R < S1.R))
		{
			// FM wins
			C.set(S3.P);
			R = S3.R;
		}
		else {
			// MiniBall wins
			C.set(S1.P);
			R = S1.R;
		}
	}
	else 
	{
		// base or FM
		if (B3 && (S3.R < S2.R))
		{
			// FM wins
			C.set(S3.P);
			R = S3.R;
		}
		else {
			// Base wins :)
			C.set(S2.P);
			R = S2.R;
		}
	}
}
