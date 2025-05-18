// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//
#pragma once

#include "../../xrCore/xrCore.h"

#pragma warning(disable:4995)
#include "directx\d3dx9.h"
#include <commctrl.h>
 
#define ENGINE_API
#define ECORE_API
#define XR_EPROPS_API

#include "..\LauncherSDL\cl_log.h"
#include "../../xrcore/clsid.h"
#include "defines.h"
#include "../../xrcdb/xrCDB.h"
#include "_d3d_extensions.h"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>

#ifdef AI_COMPILER
#	include "../../xrServerEntities/smart_cast.h"
#endif

#include "../../LauncherSDL/CompilersUI.h"
#include "../../LauncherSDL/cl_log.h"

// TODO: reference additional headers your program requires here

#define READ_IF_EXISTS(ltx,method,section,name,default_value)\
	(ltx->line_exist(section,name)) ? ltx->method(section,name) : default_value

#undef		THROW
 
IC	xr_string string2xr_string(LPCSTR s) {return s ? s : "";}
#	define	THROW(xpr)				if (!(xpr)) {throw __FILE__LINE__"\""#xpr"\"";}
#	define	THROW2(xpr,msg0)		if (!(xpr)) {throw *shared_str(xr_string(__FILE__LINE__).append(" \"").append(#xpr).append(string2xr_string(msg0)).c_str());}
#	define	THROW3(xpr,msg0,msg1)	if (!(xpr)) {throw *shared_str(xr_string(__FILE__LINE__).append(" \"").append(#xpr).append(string2xr_string(msg0)).append(", ").append(string2xr_string(msg1)).c_str());}
 