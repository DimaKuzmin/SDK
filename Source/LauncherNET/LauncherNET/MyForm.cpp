#include "MyForm.h"
#include <Windows.h>

#include "../../Editors/XrLC/xrLC.h"
 
#include "thread"
#include <vcclr.h>


using namespace System;
using namespace System::Windows::Forms;


class  NET_Logger : Logger
{
public:
    gcroot<LauncherNET::MyForm^>  form;

    void CreateClass()
    {
        Application::SetCompatibleTextRenderingDefault(false);
        Application::EnableVisualStyles();

        form = gcnew LauncherNET::MyForm();
        //LauncherNET::MyForm form;
        Application::Run(form);
    }

    void  updateLog(LPCSTR str)
    {
        form->updateLogFormItem(str);
    };

    void  updatePhrase(LPCSTR phrase)
    {
        form->updatePhaseItem(phrase);
    };
};

extern XRLC_API Logger* LoggerCL;

// Определение функции WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Application::SetCompatibleTextRenderingDefault(false);
    // Application::EnableVisualStyles();

    // LauncherNET::MyForm form;
    // Application::Run(% form);
    NET_Logger logger;

    LoggerCL = (Logger*)&logger;

    logger.CreateClass();

    return 0;
}

bool IsRunned = false;

void StartWork(SpecialArgs* args)
{
    StartupWorking("-f test  -th 16 -use_avx -use_intel -nosmg -no_invalidefaces -noise", args);
    IsRunned = false;
}

#include <msclr\marshal_cppstd.h>

System::Void LauncherNET::MyForm::button1_Click(System::Object^ sender, System::EventArgs^ e)
{
    auto Samples_str = msclr::interop::marshal_as<std::string>(Samples->Text);
    auto MUSamples_str = msclr::interop::marshal_as<std::string>(MUSamples->Text);
    auto TH_str = msclr::interop::marshal_as<std::string>(ThreadsCount->Text);
    auto PXPM_str = msclr::interop::marshal_as<std::string>(PXPM->Text);

    int _Samples = atoi(Samples_str.c_str());
    int _MUSamples = atoi(MUSamples_str.c_str());
    int _TH = atoi(TH_str.c_str());
    int _PXPM = atoi(PXPM_str.c_str());

    SpecialArgs args;
    //args.sample = ;

    args.pxpm = _PXPM;
    args.use_threads = _TH;
    args.sample = _Samples;
    args.mu_samples = _MUSamples;

    if (!IsRunned)
    {
        IsRunned = true;
        std::thread* th = new std::thread(StartWork, &args);
        th->detach();
    }
    else
    {
        LoggerCL->updateLog("Не Стартуй Не завершен еще прежний компил!!!");
    }
}