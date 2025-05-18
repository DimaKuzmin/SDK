#include "app_info.h"

CAppInfo g_AppInfo;

bool CAppInfo::IsSecondaryThread() const noexcept
{
	return false;
}

bool CAppInfo::IsPrimaryThread() const noexcept
{
	return true;
}
