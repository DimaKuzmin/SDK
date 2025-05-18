#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>
class CAppInfo
{
public:
	SDL_Window* Window = nullptr;

public:
	bool IsSecondaryThread() const noexcept;
	bool IsPrimaryThread() const noexcept;
};

extern CAppInfo g_AppInfo;