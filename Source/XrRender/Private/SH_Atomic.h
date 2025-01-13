#pragma once

#include "../../xrCore/xr_resource.h"
#include "tss_def.h"

#pragma pack(push,4)


//////////////////////////////////////////////////////////////////////////
// Atomic resources
//////////////////////////////////////////////////////////////////////////
struct ECORE_API SVS : public xr_resource_named							
{
	ID3DVertexShader*					vs;
	R_constant_table					constants;

	SVS				();
	~SVS			();
};
typedef	resptr_core<SVS,resptr_base<SVS> >	ref_vs;

//////////////////////////////////////////////////////////////////////////
struct ECORE_API SPS : public xr_resource_named
{
	ID3DPixelShader*					ps;
	R_constant_table					constants;
	~SPS			();
};
typedef	resptr_core<SPS,resptr_base<SPS> > ref_ps;

//////////////////////////////////////////////////////////////////////////
struct ECORE_API SState : public xr_resource_flagged
{
	ID3DState*							state;
	SimulatorStates						state_code;
	~SState			();
};
typedef	resptr_core<SState,resptr_base<SState> >	ref_state;

//////////////////////////////////////////////////////////////////////////
struct ECORE_API SDeclaration : public xr_resource_flagged
{
	IDirect3DVertexDeclaration9*		dcl;

	//	Use this for DirectX10 to cache DX9 declaration for comparison purpose only
	xr_vector<D3DVERTEXELEMENT9>		dcl_code;
	~SDeclaration	();
};
typedef	resptr_core<SDeclaration,resptr_base<SDeclaration> >	ref_declaration;

#pragma pack(pop)