#include "stdafx.h"

#include "uv_tri.h"
#include "xrface.h"

#include "xrLC_GlobalData.h"

bool	UVtri::similar	( const UVtri &uv, float eps/*eps = EPS*/ ) const
{
	return uv.owner == owner && _TCF::similar( uv, eps );
}