#pragma once
typedef  char string_path_ai[520];

extern string_path_ai INI_FILE;

#pragma once

#ifdef XRAI_API_EXPORTS
#	define XRAI_API __declspec(dllexport)
#else
#	define XRAI_API __declspec(dllimport)
#endif
  
void StartupAI();