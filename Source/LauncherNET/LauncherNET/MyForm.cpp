#include "pch.h"
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

void vminfo_memory(size_t* _free, size_t* reserved, size_t* committed)
{
    MEMORY_BASIC_INFORMATION memory_info;
    memory_info.BaseAddress = 0;
    *_free = *reserved = *committed = 0;
    while (VirtualQuery(memory_info.BaseAddress, &memory_info, sizeof(memory_info))) {
        switch (memory_info.State) {
        case MEM_FREE:
            *_free += memory_info.RegionSize;
            break;
        case MEM_RESERVE:
            *reserved += memory_info.RegionSize;
            break;
        case MEM_COMMIT:
            *committed += memory_info.RegionSize;
            break;
        }
        memory_info.BaseAddress = (char*)memory_info.BaseAddress + memory_info.RegionSize;
    }
}

#define Size 13
  
char* collection[Size] =
{
    "INTEL EMBREE",
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
    "MU FIRST"
};
  
void GetItemFromCollection(SpecialArgs* args, const char* item)
{
    if (strstr(item, collection[0]))
        args->use_embree = true;
    if (strstr(item, collection[1]))
        args->use_avx = true;
    if (strstr(item, collection[2]))
        args->use_sse = true;
    if (strstr(item, collection[3]))
        args->no_optimize = true;
    if (strstr(item, collection[4]))
        args->no_invalide_faces = true;
    if (strstr(item, collection[5]))
        args->nosun = true;
    if (strstr(item, collection[6]))
        args->norgb = true;
    if (strstr(item, collection[7]))
        args->nohemi = true;
    if (strstr(item, collection[8]))
        args->no_simplify = true;
    if (strstr(item, collection[9]))
        args->nosmg = true;
    if (strstr(item, collection[10]))
        args->noise = true;
    if (strstr(item, collection[11]))
        args->skip_weld = true;
    if (strstr(item, collection[12]))
        args->run_mu_first = true;
}
 

char* CvrtFloatToText(int value)
{
    char* text = new char[32];
    sprintf(text, "%f", value);
    return text;
}


char* CvrtIntToText(int value)
{
    char* text = new char[32];
    itoa(value, text, 10);
    return text;
}

gcroot<LauncherNET::MyForm^>  form;

System::Void ConvertStructure(SpecialArgs* args)
{
    if (args->EmbreeGeomType == 0)
        form->LowGeomEmbree->Checked = 1;
    else if (args->EmbreeGeomType == 1)
        form->MiddleGeomEmbree->Checked = 1;
    else if (args->EmbreeGeomType == 2)
        form->HighGeomEmbree->Checked = 1;
    else if (args->EmbreeGeomType == 3)
        form->RefitGeomEmbree->Checked = 1;

    form->EmbreeRobust->Checked = args->useRobust;
    
    if (args->LightmapSize_enum == SpecialArgs::eLightmap1024)
        form->lightmap_1024->Checked = 1;
    else if (args->LightmapSize_enum == SpecialArgs::eLightmap2048)
        form->lightmap_2048->Checked = 1;
    else if (args->LightmapSize_enum == SpecialArgs::eLightmap4096)
        form->lightmap_4096->Checked = 1;
    else if (args->LightmapSize_enum == SpecialArgs::eLightmap8192)
        form->lightmap_8192->Checked = 1;
 
      
    form->ThreadsCount->Text = gcnew System::String(CvrtIntToText(args->use_threads));
   
    form->PXPM->Text = gcnew System::String(CvrtFloatToText(args->pxpm));
    form->MUSamples->Text = gcnew System::String(CvrtIntToText(args->mu_samples));
  
     form->LevelName->Text = gcnew System::String(args->level_name.c_str());

    
    if (args->sample == 1)
        form->xrLC_JitterSamples->SelectedIndex = 0;
    else if (args->sample == 4)
        form->xrLC_JitterSamples->SelectedIndex = 1;
    else if (args->sample == 9)
        form->xrLC_JitterSamples->SelectedIndex = 2;


    // debuging 
    form->useDXT1->Checked = args->use_DXT1;

    form->FlagsCompiler->SetItemChecked(0, args->use_embree);
    form->FlagsCompiler->SetItemChecked(1, args->use_avx);
    form->FlagsCompiler->SetItemChecked(2, args->use_sse);
    form->FlagsCompiler->SetItemChecked(3, args->no_optimize);
    form->FlagsCompiler->SetItemChecked(4, args->no_invalide_faces);
    form->FlagsCompiler->SetItemChecked(5, args->nosun);
    form->FlagsCompiler->SetItemChecked(6, args->norgb);
    form->FlagsCompiler->SetItemChecked(7, args->nohemi);
    form->FlagsCompiler->SetItemChecked(8, args->no_simplify);
    form->FlagsCompiler->SetItemChecked(9, args->noise);
    form->FlagsCompiler->SetItemChecked(10, args->nosmg);
    form->FlagsCompiler->SetItemChecked(11, args->skip_weld);
    form->FlagsCompiler->SetItemChecked(12, args->run_mu_first);
 
}

 



unsigned int DeviceTime = 0;
unsigned int LAST_UPDATE = 0;

class  NET_Logger : ILogger
{
public:
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
        form->UpdateList();
    }

    virtual void  UpdateProgressBar(float value)
    {
        form->updateProgressBar(value);
    }

    virtual void UpdateTime(LPCSTR time, unsigned int time_global)
    {
        form->UpdateTime(time);
        DeviceTime = time_global;
 
        size_t reserved, free, used;
        vminfo_memory(&free, &reserved, &used);

        char text[128];
        sprintf(text, "used: %llu k\n reserved: %llu k", long int(used / 1024), long int (reserved / 1024) );
        form->UpdateMemory(text);
    }
};
  
class  NET_LoggerAI : ILoggerAI
{
public:
 
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

    virtual void UpdateProgress(float value)
    {
        form->updateProgressBar(value);

        // char string[128];
        // sprintf(string, "Progress: %f", value);
        // updateLog(string);
    }

    virtual void UpdateText()
    {
        form->UpdateList();
    }

    virtual void UpdateTime(LPCSTR time)
    {
        form->UpdateTime(time);
       
        size_t reserved, free, used;
        vminfo_memory(&reserved, &free, &used);
        char text[128];
        sprintf(text, "used memory: %llu k", long int(used / 1024));
        form->UpdateMemory(text);
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



    SpecialArgs arguments_load;
    if (LoadParrams(&arguments_load))
    {
        ConvertStructure(&arguments_load);
    }  
   
    Application::Run(form);
   
    LoggerCL = 0;
    LoggerCL_xrAI = 0;


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


    auto hitsImpl_str = msclr::interop::marshal_as < std::string >(MaxHitsCount->Text);
 
    if (lightmap_1024->Checked)
        args->LightmapSize_enum = SpecialArgs::eLightmap1024;
    else if (lightmap_2048->Checked)
        args->LightmapSize_enum = SpecialArgs::eLightmap2048;
    else if (lightmap_4096->Checked)
        args->LightmapSize_enum = SpecialArgs::eLightmap4096;
    else if (lightmap_8192->Checked)
         args->LightmapSize_enum = SpecialArgs::eLightmap8192;

    if (LowGeomEmbree->Checked)
        args->EmbreeGeomType = 0;
    else if (MiddleGeomEmbree->Checked)
        args->EmbreeGeomType = 1;
    else if (HighGeomEmbree->Checked)
        args->EmbreeGeomType = 2;
    else if (RefitGeomEmbree->Checked)
        args->EmbreeGeomType = 3;

    args->useRobust = EmbreeRobust->Checked;

    
    System::Collections::IEnumerator^ myEnum = FlagsCompiler->CheckedItems->GetEnumerator();
    while (myEnum->MoveNext())
    {
        String^ item = safe_cast<String^>(myEnum->Current);
        
        // Ваш код для обработки каждого элемента item
        String^ prefix = "Chacked: " + item;
        auto s =  msclr::interop::marshal_as<std::string>(prefix);
            GetItemFromCollection(args, s.c_str());
    };
     
    int _Samples = atoi(Samples_str.c_str());
    int _MUSamples = atoi(MUSamples_str.c_str());
    int _TH = atoi(TH_str.c_str());
    float _PXPM = atof(PXPM_str.c_str());
 

    //args->sample = ;

    args->pxpm = _PXPM;
    args->use_threads = _TH;
    args->sample = _Samples;
    args->mu_samples = _MUSamples;

    args->level_name = LevelName_str;
   
    args->adptive_ht = AdaptiveHT->Checked;
    args->cform_export = cform_export->Checked;
    args->use_DXT1 = useDXT1->Checked;
   
    args->LmapsHemi = LMAPS_HEMI_FAST->Checked;
    args->LmapsComputation = LmapsComputation->Checked;

    args->IsDOLighting = false;
 
    if (!IsRunned)
    {
        IsRunned = true;

        SaveParrams(args);

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

            sprintf(tmp, "c++: Threads: %d", args->Threads);
            LoggerCL_xrAI->updateLog(tmp);

            sprintf(tmp, "c++: Draft: %d, NoSepartor: %d, UseSpawnCompiler: %d, VerifyAI: %d, Pure Covers: :%d",
                args->Draft, args->NoSeparator, args->UseSpawnCompiler, args->VerifyAIMap, args->PureCovers);
            LoggerCL_xrAI->updateLog(tmp);

            sprintf(tmp, "c++: LevelName: %s",
                args->level_name.c_str());
            LoggerCL_xrAI->updateLog(tmp);
            sprintf(tmp, "c++: LevelOut: %s",
                args->OutSpawn_Name.c_str());
            LoggerCL_xrAI->updateLog(tmp);
            sprintf(tmp, "c++: LevelStart: %s",
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

    auto TH = msclr::interop::marshal_as<std::string>(ThreadsAI->Text);
    auto LEVEL = msclr::interop::marshal_as<std::string>(xrAI_LevelName->Text);

    args->level_name = LEVEL;
    args->Threads = atoi(TH.c_str());


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

System::Void LauncherNET::MyForm::DetailsButtonWork_Click(System::Object^ sender, System::EventArgs^ e)
{
    SpecialArgs* args = new SpecialArgs();
    args->use_threads = atoi(msclr::interop::marshal_as<std::string>(ThreadsCount_DO->Text).c_str());
    args->level_name  = msclr::interop::marshal_as < std::string >(LevelNameDO->Text).c_str();
    args->DoSamples   = atoi( msclr::interop::marshal_as<std::string>(DOSamples->Text).c_str() );
    
    args->IsDOLighting = true;
    
    if (!IsRunned)
    {
        IsRunned = true;
        StartThread(args);
    }
    else
    {
        LoggerCL->updateLog("Не Стартуй Не завершен еще прежний компил!!!");
    }

    return System::Void();
}
