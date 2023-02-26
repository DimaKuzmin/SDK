#pragma once

#include "base_color.h"


struct XRLC_LIGHT_API LightpointRequest
{
	u32 X;
	u32 Y;

	Fvector Position;
	Fvector Normal;

	void* FaceToSkip;

	LightpointRequest(u32 InX, u32 InY, Fvector InPosition, Fvector InNormal, void* InFaceToSkip)
	{
		X = InX;
		Y = InY;

		Position = InPosition;
		Normal = InNormal;

		FaceToSkip = InFaceToSkip;
	}
};





#define BORDER 1

class INetReader;
struct XRLC_LIGHT_API  lm_layer
{
	xr_vector <LightpointRequest> SurfaceLightRequests;

	u32						width;
	u32						height;
	xr_vector<base_color>	surface;
	xr_vector<u8>			marker;
private:
//	LMODE					mode;	
public:
	void					create			(u32 w, u32 h)
	{
		width				= w;
		height				= h;
		u32		size		= w*h;
		surface.clear();	surface.resize	(size);
		marker.clear();		marker.assign	(size,0);
	}
	void					destroy			()
	{
		width=height		= 0;
		surface.clear_and_free				();
		marker.clear_and_free				();
	}
	u32						Area			()						{ return (width+2*BORDER)*(height+2*BORDER); }
	void					Pixel			(u32 ID, u8& r, u8& g, u8& b, u8& s, u8& h);
	void					Pack			(xr_vector<u32>& dest)const;
	void					Pack_hemi		(xr_vector<u32>& dest)const;
	void					read			( INetReader	&r );
	void					write			( IWriter	&w ) const ;
	bool					similar			( const lm_layer &D, float eps =EPS ) const;
							lm_layer()				{ width=height=0; }

};