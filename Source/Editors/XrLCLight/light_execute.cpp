#include "stdafx.h"

#include "light_execute.h"
#include "xrdeflector.h"

void light_execute::run( CDeflector& D )
{
	D.Light	(0, &DB, &LightsSelected, H);
}

