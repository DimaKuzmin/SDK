////////////////////////////////////////////////////////////////////////////
//	Module 		: guid_generator.h
//	Created 	: 21.03.2005
//  Modified 	: 21.03.2005
//	Author		: Dmitriy Iassenev
//	Description : GUID generator
////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../../xrEngine/xrLevel.h"

#ifndef ENGINE_API 
#define ENGINE_API
#endif 

ENGINE_API extern xrGUID generate_guid();
ENGINE_API extern LPCSTR generate_guid(const xrGUID &guid, LPSTR buffer, const u32 &buffer_size);

 