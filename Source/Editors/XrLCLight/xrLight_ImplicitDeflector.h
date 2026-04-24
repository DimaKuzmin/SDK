#pragma once 

#include "lm_layer.h"
#include "xrFacedefs.h"
struct  b_BuildTexture; 

class ImplicitDeflector
{
public:
	b_BuildTexture*			texture;
	lm_layer				lmap;
	vecFace					faces;
	
	ImplicitDeflector() : texture(0)
	{
	}
	~ImplicitDeflector()
	{
		Deallocate	();
	}
	
	void			Allocate	()
	{
		lmap.create	(Width(),Height());
	}

	void			Deallocate	()
	{
		lmap.destroy();
	}
	
	u32			Width	()						;
	u32			Height	()						;	
	
	u32&		Texel	(u32 x, u32 y)			;
	base_color& Lumel	(u32 x, u32 y)			{ return lmap.surface[y*Width()+x];	}
	u8&			Marker	(u32 x, u32 y)			{ return lmap.marker [y*Width()+x];	}
	u8&			Samples(u32 x, u32 y)			{ return lmap.samples[y * Width() + x]; }

	void		Bounds			(u32 ID, Fbox2& dest);
	void		Bounds_Summary	(Fbox2& bounds);
 
};
 

#include "hash2d.h"

typedef hash2D <Face*, 384, 384>		IHASH;
class ImplicitCalcGlobs
{
	IHASH* ImplicitHash;
	ImplicitDeflector* defl;
public:
	ImplicitCalcGlobs() : defl(0) // , ImplicitHash(0)
	{

	}

	IC	IHASH& Hash()
	{
		R_ASSERT(ImplicitHash);
		return *ImplicitHash;
	}

	vecFace& query(float px, float py) { return Hash().query(px, py); };


	void Allocate()
	{
		ImplicitHash = xr_new<IHASH>();
	}

	void Deallocate()
	{
		xr_delete(ImplicitHash);
	}

	void Initialize(ImplicitDeflector& def);

	IC	ImplicitDeflector& DATA()
	{
		R_ASSERT(defl);
		return *defl;
	}
};
extern ImplicitCalcGlobs cl_globs;
