// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#pragma once

#include "optick/optick.h"
#pragma comment(lib, "OptickCore.lib")

#include "../xrLCLight/xrLC_Light.h"
#include "../../LauncherSDL/CompilersUI.h"


#define ENGINE_API				// fake, to enable sharing with engine
 
#define ECORE_API				// fake, to enable sharing with editors
#define XR_EPROPS_API
#include "../../xrcore/clsid.h"
#include "defines.h"
#include "..\LauncherSDL\cl_log.h"
 
#include "b_globals.h"
