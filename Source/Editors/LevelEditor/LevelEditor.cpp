// LevelEditor.cpp : Определяет точку входа для приложения.
//
#include "stdafx.h"
class ISE_Abstract;

#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "DbgHelp.lib")

#include "..\XrSE_Factory\xrSE_Factory_import_export.h"
extern "C"
{
    FACTORY_API	ISE_Abstract* __stdcall create_entity(LPCSTR section);
    FACTORY_API	void		__stdcall destroy_entity(ISE_Abstract*& abstract);
    FACTORY_API void		__stdcall initialize_factory();
    FACTORY_API void		__stdcall destroy_factory();
};

BOOL InitializeSymbolHandler()
{
    HANDLE hProcess = GetCurrentProcess();

    // Указываем путь к символам, например, из Microsoft Symbol Server
 
    // Инициализируем средство отладки символов
    if (SymInitialize(hProcess, 0, TRUE))
    {
        Msg("SymInitialize handler initialized successfully.\n");
        return TRUE;
    }
    else
    {
        Msg("SymInitialize failed with error: %lu\n", GetLastError());
        return FALSE;
    }
}
 

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
   bool isIntialized = InitializeSymbolHandler();

    if(!IsDebuggerPresent())
        Debug._initialize(false);  
    
    Core._initialize("Actor", ELogCallback, 1, "fs.ltx", true);
    initialize_factory();

    Tools = xr_new<CLevelTool>();
    LTools = (CLevelTool*)Tools;

    UI = xr_new<CLevelMain>();
    UI->RegisterCommands();
    LUI = (CLevelMain*)UI;

    Scene = xr_new<EScene>();


    Msg("SymInitialize: IsInitialize: %d", isIntialized);

    UIMainForm* MainForm = xr_new< UIMainForm>();
    ::MainForm = MainForm;
    UI->Push(MainForm, false);
    
    while (MainForm->Frame())
    {
    }
    xr_delete(MainForm);
    destroy_factory();
    Core._destroy();
    return 0;
}
