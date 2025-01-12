// LevelEditor.cpp : Определяет точку входа для приложения.
//
#include "stdafx.h"
class ISE_Abstract;
#include "..\XrSE_Factory\xrSE_Factory_import_export.h"
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    if(!IsDebuggerPresent())
        Debug._initialize(false);


    OPTICK_APP("Level Editor");
    OPTICK_START_THREAD("MAIN_THREAD");


    Msg("CMD START: %s", pCmdLine);
    
    Core._initialize("Actor", ELogCallback, 1, "fs.ltx", true);
    XrSE_Factory::initialize();

    Tools = xr_new<CLevelTool>();
    LTools = (CLevelTool*)Tools;

    UI = xr_new<CLevelMain>();
    UI->RegisterCommands();
    LUI = (CLevelMain*)UI;

    Scene = xr_new<EScene>();



    UIMainForm* MainForm = xr_new< UIMainForm>();
    ::MainForm = MainForm;
    UI->Push(MainForm, false);
   
    while (MainForm->Frame())
    {
    }

    OPTICK_STOP_THREAD();
    xr_delete(MainForm);
    XrSE_Factory::destroy();
    Core._destroy();
    return 0;
}
