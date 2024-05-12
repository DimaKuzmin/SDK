#include "MyForm.h"
#include <Windows.h>

#include "../../Editors/XrLC/xrLC.h"
 
#include "thread"
#include <vcclr.h>


using namespace System;
using namespace System::Windows::Forms;

#define Size 18
  
char* collection[Size] = {
    "use_embree",
    "use_avx",
    "use_sse",
    "use_opcode_old",
    "no_optimize",
    "no_invalide_faces",
    "nosun",
    "norgb",
    "nohemi",
    "no_simplify",
    "nosmg",
    "noise",
    "skip_weld",
    "use_std",
    "intel_impl",
    "intel_lmaps",
    "intel_mulight",
    "mt_calcmaterials"
};


/*
*   new 
    int embree_geometry_type = EmbreeGeom::eLow;    //++
    bool use_RobustGeom = 0;		//+
    bool skip_weld = 0;				//+
    bool use_std = 0;				//+

    // old
    bool use_embree = 0;			//+
    bool use_avx = 0;				//+
    bool use_sse = 0;				//+
    bool use_opcode_old = 0;		//+

    bool no_optimize = 0;			//+
    bool no_invalide_faces = 0;		//+

    bool nosun = 0;					//+
    bool norgb = 0;					//+
    bool nohemi = 0;				//+

    bool no_simplify = 0;			//+
    bool noise = 0;					//+
    bool nosmg = 0;					//+
         
  
*/
  
void GetItemFromCollection(SpecialArgs* args, const char* item)
{

    if (strstr(item, collection[0]))
         args->use_embree = true;
    if (strstr(item, collection[1]))
        args->use_avx = true;
    if (strstr(item, collection[2]))
        args->use_sse = true;
    if (strstr(item, collection[3]))
        args->use_opcode_old = true;
    if (strstr(item, collection[4]))
        args->no_optimize = true;
    if (strstr(item, collection[5]))
        args->no_invalide_faces = true;
    if (strstr(item, collection[6]))
        args->nosun = true;
    if (strstr(item, collection[7]))
        args->norgb = true;
    if (strstr(item, collection[8]))
        args->nohemi = true;
    if (strstr(item, collection[9]))
        args->no_simplify = true;
    if (strstr(item, collection[10]))
        args->nosmg = true;
    if (strstr(item, collection[11]))
        args->noise = true;
    if (strstr(item, collection[12]))
        args->skip_weld = true;
    if (strstr(item, collection[13]))
        args->use_std = true;

    if (strstr(item, collection[14]))
        args->use_IMPLICIT_Stage = true;
    if (strstr(item, collection[15]))
        args->use_LMAPS_Stage = true;
    if (strstr(item, collection[16]))
        args->use_MU_Lighting = true;
    if (strstr(item, collection[17]))
        args->use_mt_calculation_materials = true;

}


class  NET_Logger : ILogger
{
public:
    gcroot<LauncherNET::MyForm^>  form;

    void CreateClass()
    {
        Application::SetCompatibleTextRenderingDefault(false);
        Application::EnableVisualStyles();

        form = gcnew LauncherNET::MyForm();
         
        for (auto i = 0; i < Size; i++)
        {
            System::String^ text = gcnew System::String(collection[i]);
            form->FlagsCompiler->Items->Add(text);
        }
 
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

    virtual void  updateStatus(LPCSTR status)
    {
        form->updateStatusItem(status);
    }


    virtual void UpdateText()
    {
        form->updateALL();
    }

    virtual void UpdateTime(LPCSTR time)
    {
        form->UpdateTime(time);
    }

    virtual void updateCurrentPhase(LPCSTR str)
    {
        form->UpdateStage(str);
    }

    virtual void UpdateProgressBar(float value)
    {
       // char text[128];
       // sprintf(text, "Progress: %f" , value);
       // form->updateLogFormItem(text);
       form->UpdateProgress(value);
    }
};

extern XRLC_API ILogger* LoggerCL;

// Определение функции WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Application::SetCompatibleTextRenderingDefault(false);
    // Application::EnableVisualStyles();

    // LauncherNET::MyForm form;
    // Application::Run(% form);

//    thread_name("MAIN THREAD Aplication");
    HANDLE threadHandle = GetCurrentThread();
   
    // Устанавливаем имя потока
    SetThreadDescription(threadHandle, L"MAIN THREAD Application");


    NET_Logger logger;

    LoggerCL = (ILogger*) &logger;

    logger.CreateClass();

    return 0;
}

bool IsRunned = false;

void StartThread(SpecialArgs* main_args)
{
    std::thread* th = new std::thread(
        [] (SpecialArgs* args)
        {
            HANDLE threadHandle = GetCurrentThread();

            // Устанавливаем имя потока
            SetThreadDescription(threadHandle, L"MAIN THREAD xrLC");

            char tmp[128];
            sprintf(tmp, "c++ Arguments1: PXPM: %f, SAMPLES: %u, MUSAMPLES: %u, threads: %u, EmbreeTNear: %f", args->pxpm, args->sample, args->mu_samples, args->use_threads, args->embree_tnear);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments2: nohemi: %d, norgb: %d, nosun: %d, noise: %d, nosmg: %d", args->nohemi, args->norgb, args->nosun, args->noise, args->nosmg);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments3: no_optimize: %d, no_simplify: %d, embree: %d, avx: %d, sse: %d, use_opcode_old: %d", args->no_optimize, args->no_simplify, args->use_embree, args->use_avx, args->use_sse, args->use_opcode_old);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments4: special_flag: %s, LevelName: %s", args->special_args, args->level_name.c_str());
            LoggerCL->updateLog(tmp);

            StartupWorking(args);
            IsRunned = false;

        },
        main_args
    );
    th->detach();
}

#include <msclr\marshal_cppstd.h>



System::Void LauncherNET::MyForm::button1_Click_1(System::Object^ sender, System::EventArgs^ e)
{
    SpecialArgs* args = new SpecialArgs();

    auto Samples_str = msclr::interop::marshal_as<std::string>(Samples->Text);
    auto MUSamples_str = msclr::interop::marshal_as<std::string>(MUSamples->Text);
    auto TH_str = msclr::interop::marshal_as<std::string>(ThreadsCount->Text);
    auto PXPM_str = msclr::interop::marshal_as<std::string>(PXPM->Text);
    auto LevelName_str = msclr::interop::marshal_as < std::string >(LevelName->Text);
    auto TNear = msclr::interop::marshal_as<std::string>(EmbreeTnear->Text);

    if (RadioEmbreeGLow->Checked)
         args->embree_geometry_type = SpecialArgs::eLow;
    else if (RadioEmbreeGMedium->Checked)
        args->embree_geometry_type = SpecialArgs::eMiddle;
    else if (RadioEmbreeGHigh->Checked)
        args->embree_geometry_type = SpecialArgs::eHigh;
    else if (RadioEmbreeGUltra->Checked)
        args->embree_geometry_type = SpecialArgs::eRefit;
    
    if (RadioEmbreeG_Robust->Checked)
        args->use_RobustGeom = 1;

    args->embree_tnear = atof(TNear.c_str());

    System::Collections::IEnumerator^ myEnum = FlagsCompiler->CheckedItems->GetEnumerator();
    while (myEnum->MoveNext())
    {
        String^ item = safe_cast<String^>(myEnum->Current);
        // Ваш код для обработки каждого элемента item
        String^ prefix = "Chacked: " + item;

        auto s =  msclr::interop::marshal_as<std::string>(prefix);

        GetItemFromCollection(args, s.c_str());
     };
     
    args->off_lmaps;

    int _Samples = atoi(Samples_str.c_str());
    int _MUSamples = atoi(MUSamples_str.c_str());
    int _TH = atoi(TH_str.c_str());
    int _PXPM = atoi(PXPM_str.c_str());

    //args->sample = ;

    args->pxpm = _PXPM;
    args->use_threads = _TH;
    args->sample = _Samples;
    args->mu_samples = _MUSamples;
    args->level_name = LevelName_str;
 
    args->off_impl = off_implicit->Checked;
    args->off_lmaps = off_lmaps->Checked;
    args->off_mulitght = off_mulight->Checked;
    args->use_DXT1 = useDXT1->Checked;

    if (!IsRunned)
    {
        IsRunned = true;
         StartThread(args);
    }
    else
    {
        LoggerCL->updateLog("Не Стартуй Не завершен еще прежний компил!!!");
    }
}