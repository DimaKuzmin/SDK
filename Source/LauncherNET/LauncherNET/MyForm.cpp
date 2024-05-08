#include "MyForm.h"
#include <Windows.h>



using namespace System;
using namespace System::Windows::Forms;

 

// Определение функции WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    Application::SetCompatibleTextRenderingDefault(false);
    Application::EnableVisualStyles();

    LauncherNET::MyForm form;

    Application::Run(% form);

    return 0;
}