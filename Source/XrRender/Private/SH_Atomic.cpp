#include "stdafx.h"
#pragma hdrstop

#include "sh_atomic.h"
#include "ResourceManager.h"

#include "dxRenderDeviceRender.h"
 
///////////////////////////////////////////////////////////////////////
//	SVS
SVS::SVS() :	vs(0)
{ 
}


SVS::~SVS()
{
	DEV->_DeleteVS(this);
	_RELEASE(vs);
}


///////////////////////////////////////////////////////////////////////
//	SPS
SPS::~SPS								()			{	_RELEASE(ps);		DEV->_DeletePS			(this);	}

//	SState
SState::~SState							()			{	_RELEASE(state);	DEV->_DeleteState		(this);	}

///////////////////////////////////////////////////////////////////////
//	SDeclaration
SDeclaration::~SDeclaration()
{	
	DEV->_DeleteDecl(this);	
	_RELEASE(dcl);
}
