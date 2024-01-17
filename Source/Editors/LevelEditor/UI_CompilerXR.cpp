#include "stdafx.h"
#include "UI_CompilerXR.h"

UI_CompilerXR* UI_CompilerXR::Form = nullptr;
bool UI_CompilerXR::reinit = false;

void UI_CompilerXR::Show()
{
	if (Form == nullptr)
	{	
		Form = xr_new< UI_CompilerXR>();
 
		string_path path;
		FS.update_path(path, "$fs_root$", "_appdata_\\CompilersXR.cfg");
	 
		if (FS.exist(path))
		{
			//CInifile* file= xr_new<CInifile>(path);
			//Form->ReadLTX(file);
			IReader* read = FS.r_open(path);
			Form->ReadBytes(read);
			Msg("Load Config: %s", path);
			
		}
	}
}

void UI_CompilerXR::Close()
{
	xr_delete(Form);
}

void UI_CompilerXR::Update()
{ 
	if (Form)
	{
		if (!Form->IsClosed())
		{
			Form->Draw();
		}
		else
		{
			reinit = false;
			xr_delete(Form);
		}
	}
}




void UI_CompilerXR::Draw()
{
	if (!ImGui::Begin("Compilers Params", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}

	"-? or -h			== this help\n"
	"-modify_options	== modify build options\n"
	"-nosun				== disable sun-lighting\n"
	"-nosmg				== disable smoth groops\n"		    
	
	//NEW SDK 8.0
	"-no_invalidefaces  == OFF invalidate faces\n"
	"-no_optimize		== OFF optimize geometry\n"
//Se7Kills
	"\n"
	"New Se7Kills Flags: \n"
	"-norgb		== Отключить статику\n"		 
	"-nohemi	== Отключить Hemi\n"		 

 	//NEW THREADING AND CUSTOMIZE PARAMS
	"-pxpm		== пиксели из SDK\n"
	"-sample 	== колво сэмполов из SDK (1, 4, 9)\n"
	"-mu_samples== колво сэмполов для mu моделей (1-6) \n"
	"-th (num)	== кол-во потоков для IMPLICIT, LMAPS \n"
	"-no_simplify == не делать Упрощение CFORM (Иногда по памяти затратно)\n "
	
	"\n"
	"Новая Компиляция света IntelEmbree:\n"
	"-use_intel	== Включить IntelEmbree \n"
	"-use_avx	== Включить AVX \n" 
	"-use_sse	== Включить SSE \n"
 	"-hw_light  == Не допилено\n"
	"-use_opcode_old == использовать не оптимизированый OPCODE\n"

	"\n"
	"Пример: start bin\\x64\\dev\\xrLC.exe -nosmg -f jupiter -no_invalidefaces -no_simplify -noise -th 16 -sample 9 -mu_samples 6 -use_intel -use_avx \n"
 	
	"\n"
	"Важный параметр:\n"
	"-f<NAME>	== compile level in GameData\\Levels\\<NAME>\\\n" 
	"\n"
	"NOTE: The last key is required for any functionality\n";

	bool modified = false;

	modified |= ImGui::Checkbox("nosun", &nosun);
	modified |= ImGui::Checkbox("nohemi", &nohemi);
	modified |= ImGui::Checkbox("norgb", &norgb);

	modified |= ImGui::Checkbox("noise", &noise);

	modified |= ImGui::Checkbox("no_simplify", &no_simplify);
	modified |= ImGui::Checkbox("no_invalidatefaces", &no_invalidatefaces);
	modified |= ImGui::Checkbox("no_optimize", &no_optimize);

	modified |= ImGui::Checkbox("hw_light", &hw_light);

	modified |= ImGui::Checkbox("use_intel", &use_intel);
	modified |= ImGui::Checkbox("use_avx", &use_avx);
	modified |= ImGui::Checkbox("use_sse", &use_sse);

	modified |= ImGui::InputInt("pxpm", &pxpm);
	modified |= ImGui::InputInt("mu-samples", &mu_samples);
	modified |= ImGui::InputInt("sample", &sample );
	modified |= ImGui::InputInt("th", &th);



	string256 start_params = {0};

	
	if (use_intel)
		xr_strcat(start_params, "-use_intel ");
	if (use_avx)
		xr_strcat(start_params, "-use_avx ");
	if (use_sse)
		xr_strcat(start_params, "-use_sse ");

	string16 tmp;
	sprintf(tmp, "-th %d ", th);
	xr_strcat(start_params, tmp);


	if (ImGui::Button("Start xrLC")) 
	{

		if (noise)
			xr_strcat(start_params, "-noise ");
		if (norgb)
			xr_strcat(start_params, "-norgb ");
		if (nosun)
			xr_strcat(start_params, "-nosun ");
		if (nohemi)
			xr_strcat(start_params, "-nohemi ");
		if (no_simplify)
			xr_strcat(start_params, "-no_simplify ");
		if (no_invalidatefaces)
			xr_strcat(start_params, "-no_invalidatefaces ");
		if (no_optimize)
			xr_strcat(start_params, "-no_optimize ");
		
		if (hw_light)
			xr_strcat(start_params, "-hw_light ");
		
		sprintf(tmp, "-pxpm %d ", pxpm);
 		xr_strcat(start_params, tmp);
		
		sprintf(tmp, "-mu_samples %d ", mu_samples);
 		xr_strcat(start_params, tmp);

		sprintf(tmp, "-sample %d ", sample);
 		xr_strcat(start_params, tmp);

		
		sprintf(tmp, "-f %s", Scene->m_LevelOp.m_FNLevelPath.c_str());
		xr_strcat(start_params, tmp);

		 
		string_path file;
		FS.update_path(file, "$fs_root$", "");

		xr_strcat(file, "Bin\\x64\\Compilers_x64\\xrLC.exe ");
		//xr_strcat(file, start_params);		

		Msg("Start: %s, Args: %s", file, start_params);

		//std::system(file);
		 
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));


		CreateProcess(file, start_params, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
		
		
		//LPSTR buffer;
		//GetModuleFileName(NULL, buffer, MAX_PATH);
		//int result = std::system(appPath);

		//Msg("Root: %s", buffer);
	}

	modified |= ImGui::Checkbox("Draft aimap", &draft_aimap);
	modified |= ImGui::Checkbox("NO Separator Check", &no_separator_check);
	
	//modified |= ImGui::InputText("SPAWN Levels: ", levels_spawn, sizeof(levels_spawn));
	//modified |= ImGui::InputText("SPAWN Levels Out: ", levels_out, sizeof(levels_out));

	if (ImGui::Button("Start xrAI (ai-map)"))
	{
		if (draft_aimap)
			xr_strcat(start_params, "-draft ");

		
		sprintf(tmp, "-f %s", Scene->m_LevelOp.m_FNLevelPath.c_str());
		xr_strcat(start_params, tmp);


		string_path file;
		FS.update_path(file, "$fs_root$", "");

		xr_strcat(file, "Bin\\x64\\Compilers_x64\\xrAI.exe ");
 
		Msg("Start: %s, Args: %s", file, start_params);
	
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));


		CreateProcess(file, start_params, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
	}

	if (ImGui::Button("Start xrAI (current level alife)"))
	{
		char* args = "-out alife -no_separator_check";
		string256 start_params_new = {0};
		sprintf(start_params_new, "%s -s %s", args,  Scene->m_LevelOp.m_FNLevelPath.c_str() );
  
		
		string_path file;
		FS.update_path(file, "$fs_root$", "");
		xr_strcat(file, "Bin\\x64\\Compilers_x64\\xrAI.exe ");
 
	
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));

		CreateProcess(file, start_params_new, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
	}

	/*
	if (ImGui::Button("Start xrAI (spawn)"))
	{	
		if (no_separator_check)
			xr_strcat(start_params, "-no_separator_check ");

		sprintf(tmp, "-s %s ", levels_spawn);
		xr_strcat(start_params, tmp);

		sprintf(tmp, "-out %s ", levels_out);
		xr_strcat(start_params, tmp);

		string_path file;
		FS.update_path(file, "$fs_root$", "");

		xr_strcat(file, "Bin\\x64\\Compilers_x64\\xrAI.exe ");
 
		Msg("Start: %s, Args: %s", file, start_params);
	
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));


		CreateProcess(file, start_params, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
	}
	*/

	ImGui::End();


	if (modified)
	{
		string_path filep;
		FS.update_path(filep, "$fs_root$", "_appdata_\\CompilersXR.cfg");
	 
 		/*
		CInifile* file = xr_new<CInifile>(filep, false, false, true);
		if (file)
		   WriteLTX(file);
		file->save_as();
		Msg("File Save: %s", filep);
		*/

		IWriter* w = FS.w_open(filep);
		WriteBytes(w);
		FS.w_close(w);
	}
}

void UI_CompilerXR::ReadLTX(CInifile* file)
{
	if (!file)
		return;

	nosun = file->r_bool("compiler_parrams", "nosun");
	nohemi = file->r_bool("compiler_parrams", "nohemi");
	norgb = file->r_bool("compiler_parrams", "norgb");

	noise = file->r_bool("compiler_parrams", "noise");
	no_simplify = file->r_bool("compiler_parrams", "no_simplify");
	no_invalidatefaces = file->r_bool("compiler_parrams", "no_invalidatefaces");
				
	no_optimize = file->r_bool("compiler_parrams", "no_optimize");

	hw_light	= file->r_bool("compiler_parrams", "hw_light");
	use_intel	= file->r_bool("compiler_parrams", "use_intel");
	use_avx = file->r_bool("compiler_parrams", "use_avx");
	use_sse = file->r_bool("compiler_parrams", "use_sse");

	hw_light = file->r_u32("compiler_parrams", "pxpm");
	mu_samples = file->r_u32("compiler_parrams", "mu_samples");
	sample = file->r_u32("compiler_parrams", "sample");
	th = file->r_u32("compiler_parrams", "th");

	draft_aimap = file->r_bool("compiler_parrams", "draft_aimap");
	no_separator_check = file->r_bool("compiler_parrams", "no_separator_check");
 
	xr_strcpy(levels_spawn, file->r_string("compiler_parrams", "levels_spawn")); 
	xr_strcpy(levels_out,   file->r_string("compiler_parrams", "levels_out"));
 }

void UI_CompilerXR::WriteLTX(CInifile* file)
{
	if (!file)
		return;

	file->w_bool("compiler_parrams", "nosun", nosun);
	file->w_bool("compiler_parrams", "nohemi", nohemi);
	file->w_bool("compiler_parrams", "norgb", norgb);

	file->w_bool("compiler_parrams", "noise", noise);
	file->w_bool("compiler_parrams", "no_simplify", no_simplify);
	file->w_bool("compiler_parrams", "no_invalidatefaces", no_invalidatefaces);

	file->w_bool("compiler_parrams", "no_optimize", no_optimize);

	file->w_bool("compiler_parrams", "hw_light", hw_light);
	file->w_bool("compiler_parrams", "use_intel", use_intel);
	file->w_bool("compiler_parrams", "use_avx", use_avx);
	file->w_bool("compiler_parrams", "use_sse", use_sse);

	file->w_u32("compiler_parrams", "pxpm", hw_light);
	file->w_u32("compiler_parrams", "mu_samples", mu_samples);
	file->w_u32("compiler_parrams", "sample", sample);
	file->w_u32("compiler_parrams", "th", th);

	file->w_bool("compiler_parrams", "draft_aimap", draft_aimap);
	file->w_bool("compiler_parrams", "no_separator_check", no_separator_check);
	 
			
	file->w_string("compiler_parrams", "levels_spawn", levels_spawn);
	file->w_string("compiler_parrams", "levels_out",   levels_out);
}

 

void UI_CompilerXR::WriteBytes(IWriter* file)
{
	if (!file)
		return;

	file->w_u8(nosun);
	file->w_u8(norgb);
	file->w_u8(nohemi);
	file->w_u8(no_invalidatefaces);
	file->w_u8(no_optimize);
 
	file->w_u8(no_simplify);
	file->w_u8(use_intel);
	file->w_u8(use_avx);
	file->w_u8(use_sse);

	file->w_u8(hw_light);
	file->w_u8(use_opcode_older);
 
	file->w_u8(pxpm);
	file->w_u8(sample);
	file->w_u8(mu_samples);
	file->w_u8(th);
 
	file->w_u8(noise);
	file->w_u8(draft_aimap);
	file->w_u8(no_separator_check);
 
	file->w_string(levels_spawn);
	file->w_string(levels_out);

}

void UI_CompilerXR::ReadBytes(IReader* file)
{
	if (!file)
		return;

	nosun = file->r_u8();
	norgb = file->r_u8();
	nohemi = file->r_u8();
	no_invalidatefaces = file->r_u8();
	no_optimize = file->r_u8();
 
	no_simplify = file->r_u8();
	use_intel = file->r_u8();
	use_avx = file->r_u8();
	use_sse = file->r_u8();

	hw_light = file->r_u8();
	use_opcode_older = file->r_u8();
 
	pxpm = file->r_u8();
	sample = file->r_u8();
	mu_samples = file->r_u8();
	th = file->r_u8();
 
	noise = file->r_u8();
	draft_aimap = file->r_u8();
	no_separator_check = file->r_u8();

	{
		xr_string s;
		file->r_string(s); 
		xr_strcpy(levels_spawn, s.c_str());
	}

	
	{
		xr_string s;
		file->r_string(s); 
		xr_strcpy(levels_out, s.c_str());
	}
 
}

  