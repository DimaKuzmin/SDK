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

void				OGF_Base::CalcBounds	() 
{
	MSSphereV1 = 0;
	MSSphereV2 = 0;
	MSSphereV3 = 0;
	MSVALIDATION = 0;

	// get geometry
	thread_local xr_vector<Fvector>		V;
	V.clear						();
	V.reserve					(4096);
	GetGeometry					(V);
	 
	R_ASSERT					(V.size() >= 3); 

	// Se7kills 
	CTimer t; t.Start();

	// 1: calc first variation
	Fsphere	S1;
	Fsphere_compute				(S1,&*V.begin(),(u32)V.size());
	
	MSSphereV1 += t.GetElapsed_ms(); t.Start();
 
	// 2: calc ordinary algorithm (2nd)
	Fsphere S2;
	Sphere_Compute(bbox, V, S2);

	MSSphereV2 += t.GetElapsed_ms(); t.Start();

	// 3: calc magic-fm
	Fsphere S3;
	SphereComputeMagic(V, S3);
	
	MSSphereV3 += t.GetElapsed_ms(); t.Start();

	BOOL B1 = SphereValid(V, S1);
	BOOL B2 = SphereValid(V, S2);
	BOOL B3 = SphereValid(V, S3);

	MSVALIDATION += t.GetElapsed_ms();

	// select best one
	if (B1 && (S1.R<S2.R))
	{
		// miniball or FM
		if (B3 && (S3.R<S1.R))
		{
			// FM wins
			C.set	(S3.P);
			R	=	S3.R;
		} else {
			// MiniBall wins
			C.set	(S1.P);
			R	=	S1.R;
		}
	}
	else
	{
		// base or FM
		if (B3 && (S3.R<S2.R))
		{
			// FM wins
			C.set	(S3.P);
			R	=	S3.R;
		} else {
			// Base wins :)
			C.set	(S2.P);
			R	=	S2.R;
		}
	}
}
