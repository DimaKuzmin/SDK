#include "MyForm.h"
#include <Windows.h>

#include "../../Editors/XrLC/xrLC.h"
#include "../../Editors/XrAI/xrAI.h"

#include "thread"
#include <vcclr.h>

#pragma comment(lib, "xrLC.lib")
#pragma comment(lib, "xrAI.lib")


using namespace System;
using namespace System::Windows::Forms;

#define Size 15
  
char* collection[Size] =
{
    "AVX INSTRUCTIONS",
    "SSE INSTRUCTIONS",
    "NO OPTIMIZE",
    "SKIP INVALID",
    "NO SUN",
    "NO RGB",
    "NO HEMI",
    "NO SIMPLIFY",
    "NO SMG",
    "NOISE GEOM",
    "SKIP WELD",
    "USE STD THREADS",
    "INTEL IMPLICIT",
    "INTEL LMAPS",
    "INTEL MU MODELS"
};
  
void GetItemFromCollection(SpecialArgs* args, const char* item)
{
    if (strstr(item, collection[0]))
        args->use_avx = true;
    if (strstr(item, collection[1]))
        args->use_sse = true;
    if (strstr(item, collection[2]))
        args->no_optimize = true;
    if (strstr(item, collection[3]))
        args->no_invalide_faces = true;
    if (strstr(item, collection[4]))
        args->nosun = true;
    if (strstr(item, collection[5]))
        args->norgb = true;
    if (strstr(item, collection[6]))
        args->nohemi = true;
    if (strstr(item, collection[7]))
        args->no_simplify = true;
    if (strstr(item, collection[8]))
        args->nosmg = true;
    if (strstr(item, collection[9]))
        args->noise = true;
    if (strstr(item, collection[10]))
        args->skip_weld = true;
    if (strstr(item, collection[11]))
        args->use_std = true;

    if (strstr(item, collection[12]))
        args->use_IMPLICIT_Stage = true;
    if (strstr(item, collection[13]))
        args->use_LMAPS_Stage = true;
    if (strstr(item, collection[14]))
        args->use_MU_Lighting = true;

   

}
#include <vcclr.h> // Include for gcroot

gcroot<LauncherNET::MyForm^>  form;

class  NET_Logger : ILogger
{
public:
    void  updateLog(LPCSTR str)
    {
        if (form.operator->() != nullptr)
            form->updateLogFormItem(str);
        else
            DebugBreak();
    };

    void  updatePhrase(LPCSTR phrase)
    {
        if (form.operator->() != nullptr)
            form->updatePhaseItem(phrase);
        else
            DebugBreak();
    };

    virtual void  updateStatus(LPCSTR status)
    {
        if (form.operator->() != nullptr)
            form->updateStatusItem(status);
        else
            DebugBreak();
     }


    virtual void UpdateText()
    {
        if (form.operator->() != nullptr)
            form->updateALL();
        else
            DebugBreak();
        
    }

    virtual void UpdateTime(LPCSTR time)
    {
        if (form.operator->() != nullptr)
            form->UpdateTime(time);
        else
            DebugBreak();
       
    }
};

class  NET_LoggerAI : ILoggerAI
{
public:
    void  updateLog(LPCSTR str)
    {
        if (form.operator->() != nullptr)
            form->updateLogFormItem(str);
        else
            DebugBreak();
    };

    void  updatePhrase(LPCSTR phrase)
    {
        if (form.operator->() != nullptr)
            form->updatePhaseItem(phrase);
        else
            DebugBreak();
    };

    virtual void  updateStatus(LPCSTR status)
    {
        if (form.operator->() != nullptr)
            form->updateStatusItem(status);
        else
            DebugBreak();
    }


    virtual void UpdateText()
    {
        if (form.operator->() != nullptr)
            form->updateALL();
        else
            DebugBreak();

    }

    virtual void UpdateTime(LPCSTR time)
    {
        if (form.operator->() != nullptr)
            form->UpdateTime(time);
        else
            DebugBreak();

    }
};

extern XRLC_API ILogger* LoggerCL;
extern XRAI_API ILoggerAI* LoggerCL_xrAI;


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

    NET_LoggerAI lAI;
    LoggerCL_xrAI = (ILoggerAI*)&lAI;

    NET_Logger lLC;
    LoggerCL = (ILogger*)&lLC;


    Application::SetCompatibleTextRenderingDefault(false);
    Application::EnableVisualStyles();

    form = gcnew LauncherNET::MyForm();

    for (auto i = 0; i < Size; i++)
    {
        System::String^ text = gcnew System::String(collection[i]);
        form->FlagsCompiler->Items->Add(text);
    }

    form->xrLC_JitterSamples->MaxDropDownItems = 3;
    form->xrLC_JitterSamples->SelectedIndex = 0;

   
    Application::Run(form);
   

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
            /*
            char tmp[128];
            sprintf(tmp, "c++ Arguments1: PXPM: %f, SAMPLES: %u, MUSAMPLES: %u, threads: %u, EmbreeTNear: %f", args->pxpm, args->sample, args->mu_samples, args->use_threads, args->embree_tnear);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments2: nohemi: %d, norgb: %d, nosun: %d, noise: %d, nosmg: %d", args->nohemi, args->norgb, args->nosun, args->noise, args->nosmg);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments3: no_optimize: %d, no_simplify: %d, embree: %d, avx: %d, sse: %d, use_opcode_old: %d", args->no_optimize, args->no_simplify, args->use_embree, args->use_avx, args->use_sse, args->use_opcode_old);
            LoggerCL->updateLog(tmp);

            sprintf(tmp, "c++ Arguments4: special_flag: %s, LevelName: %s", args->special_args, args->level_name.c_str());
            LoggerCL->updateLog(tmp);
            */
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

    auto Samples_str = msclr::interop::marshal_as<std::string>(xrLC_JitterSamples->GetItemText(xrLC_JitterSamples->SelectedItem));
    auto MUSamples_str = msclr::interop::marshal_as<std::string>(MUSamples->Text);
    auto TH_str = msclr::interop::marshal_as<std::string>(ThreadsCount->Text);
    auto PXPM_str = msclr::interop::marshal_as<std::string>(PXPM->Text);
    auto LevelName_str = msclr::interop::marshal_as < std::string >(LevelName->Text);
    auto TNear = msclr::interop::marshal_as<std::string>(EmbreeTnear->Text);
    auto HITS_str = msclr::interop::marshal_as<std::string>(EmbreeHitsCollect->Text);

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

    args->use_embree = args->use_IMPLICIT_Stage || args->use_LMAPS_Stage || args->use_MU_Lighting;
     
    args->off_lmaps;

    int _Samples = atoi(Samples_str.c_str());
    int _MUSamples = atoi(MUSamples_str.c_str());
    int _TH = atoi(TH_str.c_str());
    int _PXPM = atoi(PXPM_str.c_str());
    int _HITS = atoi( HITS_str.c_str() );

    //args->sample = ;

    args->pxpm = _PXPM;
    args->use_threads = _TH;
    args->sample = _Samples;
    args->mu_samples = _MUSamples;
    args->MaxHitsPerRay = _HITS;


    args->level_name = LevelName_str;
 
    args->off_impl = off_implicit->Checked;
    args->off_lmaps = off_lmaps->Checked;
    args->off_mulitght = off_mulight->Checked;
    args->use_DXT1 = useDXT1->Checked;
    args->precalc_triangles = use_PrecalcTris->Checked;

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

void StartThread_xrAI(SpecialArgsAI* argsb)
{
    std::thread* th = new std::thread(
        [](SpecialArgsAI* args)
        {
            HANDLE threadHandle = GetCurrentThread();

            // Устанавливаем имя потока
            SetThreadDescription(threadHandle, L"MAIN THREAD xrAI");

            char tmp[128];
            sprintf(tmp, "c++ Arguments: Draft: %d, NoSepartor: %d, UseSpawnCompiler: %d, VerifyAI: %d",
                args->Draft, args->NoSeparator, args->PureCovers, args->UseSpawnCompiler, args->VerifyAIMap);
            LoggerCL_xrAI->updateLog(tmp);

            sprintf(tmp, "c++ Arguments: LevelName: %s",
                args->level_name.c_str());
            LoggerCL_xrAI->updateLog(tmp);
            sprintf(tmp, "c++ Arguments: LevelOut: %s",
                args->OutSpawn_Name.c_str());
            LoggerCL_xrAI->updateLog(tmp);
            sprintf(tmp, "c++ Arguments: LevelStart: %s",
                args->SpawnActorStart.c_str());
            LoggerCL_xrAI->updateLog(tmp);


            StartupWorking_xrAI(args);
            IsRunned = false;

        }, argsb
    );
}


System::Void LauncherNET::MyForm::xrAI_SpawnAIMap_Click(System::Object^ sender, System::EventArgs^ e)
{
    SpecialArgsAI* args = new SpecialArgsAI();

    args->Draft = xrAI_Draft->Checked;
    args->PureCovers = xrAI_PureCovers->Checked;
    args->UseSpawnCompiler = false;
    args->VerifyAIMap = xrAI_Verify->Checked;


    auto LEVEL = msclr::interop::marshal_as<std::string>(xrAI_LevelName->Text);

    args->level_name = LEVEL;
    
    if (!IsRunned)
    {
        StartThread_xrAI(args);
    }
    else
    {
        LoggerCL_xrAI->updateLog("Не Стартуй Не завершен еще прежний компил!!!");
    }
}

System::Void LauncherNET::MyForm::xrAI_StartSpawn_Click(System::Object^ sender, System::EventArgs^ e)
{
    SpecialArgsAI* args = new SpecialArgsAI();

    args->NoSeparator = xrAI_NoSepartor->Checked;
    args->UseSpawnCompiler = true;
    

    auto LEVEL = msclr::interop::marshal_as<std::string>(xrAI_LevelsName_Spawn->Text);
    auto OUTSPAWN = msclr::interop::marshal_as<std::string>(xrAI_SpawnOut->Text);
    auto START = msclr::interop::marshal_as<std::string>(xrAI_SPStartLevel->Text);

    args->level_name = LEVEL;
    args->OutSpawn_Name = OUTSPAWN;
    args->SpawnActorStart = START;


    if (!IsRunned)
    {
        StartThread_xrAI(args);
    }
    else
    {
        LoggerCL_xrAI->updateLog("Не Стартуй Не завершен еще прежний компил!!!");
    }
}
