// LevelEditor.cpp : Определяет точку входа для приложения.
//
#include "stdafx.h"
class ISE_Abstract;

#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "DbgHelp.lib")


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
 
#include "..\XrSE_Factory\xrSE_Factory_import_export.h"
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
   bool isIntialized = InitializeSymbolHandler();

    if(!IsDebuggerPresent())
        Debug._initialize(false);

    OPTICK_APP("Level Editor");

    OPTICK_STOP_CAPTURE();

    OPTICK_START_THREAD("MAIN_THREAD");

    //   OPTICK_CATEGORY("CategoryName", Optick::Category::Scene);



    Msg("CMD START: %s", pCmdLine);
    
    Core._initialize("Actor", ELogCallback, 1, "fs.ltx", true);
    XrSE_Factory::initialize();

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
    XrSE_Factory::destroy();
    Core._destroy();
    return 0;
}
