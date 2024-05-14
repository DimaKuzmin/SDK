#pragma once
typedef  char string_path_ai[520];

extern string_path_ai INI_FILE;

#pragma once

#ifdef XRAI_API_EXPORTS
#	define XRAI_API __declspec(dllexport)
#else
#	define XRAI_API __declspec(dllimport)
#endif

#include <string>

struct XRAI_API  SpecialArgsAI
{
	std::string level_name; // OrLevels
	std::string OutSpawn_Name;
	std::string SpawnActorStart;

	// SELECT COMPILER
	bool UseSpawnCompiler;


	// AI MAP
	bool Draft;
	bool PureCovers;
	bool VerifyAIMap;

	//SPAWN
	bool NoSeparator; 
}; 


XRAI_API void  StartupWorking_xrAI(SpecialArgsAI* args);

class XRAI_API ILoggerAI
{
public:
	virtual void  updateLog(LPCSTR str) = 0;
	virtual void  updatePhrase(LPCSTR phrase) = 0;
	virtual void  updateStatus(LPCSTR status) = 0;

	virtual void  UpdateText() = 0;
	virtual void  UpdateTime(LPCSTR time) = 0;
};

extern XRAI_API ILoggerAI* LoggerCL_xrAI;