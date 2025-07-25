#include "app_info.h"

#pragma comment(lib, "Luabind.lib")
#pragma comment(lib, "lua51.lib")
#pragma comment(lib, "winmm.lib") 

#pragma comment(lib, "d3dx9.lib")
#pragma comment(lib, "SDL3.lib")
#pragma comment(lib, "FreeMagic.lib")
#pragma comment(lib, "BearCore.lib")
#pragma comment(lib, "BearGraphics.lib")

// Xray
#pragma comment(lib, "xrCore.lib")
#pragma comment(lib, "xrCDB.lib")

#pragma comment(lib, "xrLCLight.lib")
#pragma comment(lib, "xrLC.lib")
#pragma comment(lib, "xrAI.lib")
 

#pragma comment(lib, "xrDXT.lib")
 
CAppInfo g_AppInfo;

bool CAppInfo::IsSecondaryThread() const noexcept
{
	return false;
}

bool CAppInfo::IsPrimaryThread() const noexcept
{
	return true;
}
