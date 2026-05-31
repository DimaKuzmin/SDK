#include "CompilersUI.h"
#include "cl_log.h"
#include <timeapi.h>
#include "imgui/imgui.h"
#include "app_info.h"
#include <Psapi.h>
#include "ImGUI_Style.h"

//Ex: 25, 200, 50, 255 -> 0.0980392, 0.784314, 0.196078, 1
#define RGBAColor(r,g,b,a) r/(float)255, g/(float)255, b/(float)255, a/(float)255

extern CompilersMode gCompilerMode;
 
void InitializeUIData()
{
	string_path LevelsDir = {};
	FS.update_path(LevelsDir, "$game_levels$", "");

	FS_FileSet FSLIST;
	FS.file_list(FSLIST, LevelsDir, FS_ListFolders | FS_RootOnly);

	for (auto FOLDER : FSLIST)
	{
 		gCompilerMode.Files.emplace_back().Name = FOLDER.name;

		u32 s = gCompilerMode.Files.back().Name.size();
		gCompilerMode.Files.back().Name.resize(s-1);
	}
}

void DrawCompilerConfig();
void DrawAIConfig();
void DrawDOConfig();
void DrawLCConfig();

static bool autoScroll = true;
bool ShowMainUI = true;

void DrawBottonUI()
{
	if (ImGui::Button(autoScroll ? "Disable Auto-Scroll" : "Enable Auto-Scroll"))
		autoScroll = !autoScroll;
	ImGui::SameLine();
	ImGui::Checkbox("SwitchUI", &ShowMainUI);
	ImGui::SameLine();
	ImGui::TextColored(ImVec4(172, 172, 255, 255), "Memory: %u mb", GetHeapMemory() / 1024 / 1024);
	ImGui::SameLine();
	if (ImGui::Button("SwitchTheme"))
	{
		if (CIMStyle.isRedTheme)
			CIMStyle.BlackTheme();
		else
			CIMStyle.RedTheme();
	}
}

void RenderMainUI()
{
 	Uint32 flags = SDL_GetWindowFlags(g_AppInfo.Window);
	bool is_minimized = (flags & SDL_WINDOW_MINIMIZED) != 0;

	if (is_minimized)		return;

	int Size[2] = {};
	SDL_GetWindowSize(g_AppInfo.Window, &Size[0], &Size[1]);
	ImGui::SetNextWindowPos({ 0, 0 });
	ImGui::SetNextWindowSize({ (float)Size[0], (float)Size[1] });
	 
	if (!ShowMainUI)
	{
		RenderCompilerUI(Size[0], Size[1]);
		return;
	}

	if (Size[0] != 1000 || Size[1] != 540)
	{
		SDL_SetWindowSize(g_AppInfo.Window, 1000, 600);
	}

	if (ImGui::Begin("MainForm", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus))
	{
		u32 WindowSizeY = 100;
		ImVec2 ListBoxSize = { float(Size[0] - 20), float(Size[1] - WindowSizeY) };
		if (ImGui::BeginTable("##Levels", 5, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY, ListBoxSize))
		{
			// 
			ImGui::TableSetupColumn("Levels");
			ImGui::TableSetupColumn("Settings");
			ImGui::TableSetupColumn("xrLC");
			ImGui::TableSetupColumn("xrAI");
			ImGui::TableSetupColumn("xrDO");

			ImGui::TableHeadersRow();

			ImGui::TableNextRow();
			ImGui::TableNextColumn();

			ImVec2 ListBoxSize2 = { 200, float(Size[1] - (WindowSizeY + 40) ) };
			if (ImGui::BeginTable("##Levels", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY, ListBoxSize2))
			{
				ImGui::TableSetupColumn("Name");
				ImGui::TableSetupColumn("Prop");
				ImGui::TableHeadersRow();

				size_t Iter = 1;
				for (auto& [File, Selected] : gCompilerMode.Files)
				{
					ImGui::TableNextColumn();
					// xr_string U8Str = Platform::ANSI_TO_UTF8(File);
					ImGui::Selectable(File.c_str());

					ImGui::TableNextColumn();

 					ImGui::Checkbox(("##check" + File).c_str(), &Selected);
					Iter++;

					if (Iter < gCompilerMode.Files.size())
					{
						ImGui::TableNextRow();
					}
				}
				ImGui::EndTable();
			}

			ImGui::TableNextColumn();

			DrawCompilerConfig();

			ImGui::TableNextColumn();

			DrawLCConfig();

			ImGui::TableNextColumn();

			DrawAIConfig();

			ImGui::TableNextColumn();

			DrawDOConfig();

			ImGui::EndTable();
		}
	}
	
	auto BSize = ImGui::GetContentRegionAvail();
 	if (ImGui::Button("Run Compiler", { BSize.x, 50 }))
	{
		bool isReady = gCompilerMode.LC || gCompilerMode.DO;
 		if (gCompilerMode.AI) 
			isReady = gCompilerMode.AI_BuildLevel || gCompilerMode.AI_BuildSpawn;
 
		if (isReady)
		{
			for (auto& FILE : gCompilerMode.Files)
			{
				if (FILE.Select)
				{
					Msg("Level For Building : %s", FILE.Name);
					break;
				}
			}

			ShowMainUI = false;
			extern void StartCompile();
			StartCompile();
		}

	}

	DrawBottonUI();
	ImGui::End();
}


const ImVec4 getLogColor_new(char* text)
{
	if (text == nullptr || xr_strlen(text) == 0)
		return ImVec4(RGBAColor(230, 230, 230, 255));

	xr_string TextEx = text;
	TextEx = TextEx.RemoveWhitespaces();
	size_t Pos = TextEx.find('|');

	while (Pos != xr_string::npos)
	{
		TextEx.erase(Pos, 1);
		Pos = TextEx.find('|');
	}

	char Word = TextEx[0];

	switch (Word)
	{
		case '~': return ImVec4(RGBAColor(248, 248, 49, 255));
		case '!': return ImVec4(RGBAColor(204, 102, 102, 255));
		case '@': return ImVec4(RGBAColor(125, 125, 241, 255));
		case '#': return ImVec4(RGBAColor(0, 222, 205, 155));
		case '%': return ImVec4(RGBAColor(202, 85, 219, 155));
		case '$': return ImVec4(RGBAColor(172, 172, 255, 255));
		case '*': return ImVec4(RGBAColor(248, 248, 49, 255));
		case '^': return ImVec4(RGBAColor(100, 246, 121, 255));
		case '&': return ImVec4(RGBAColor(255, 255, 0, 255));
		case '-': return ImVec4(RGBAColor(0, 255, 0, 255));
		case '+': return ImVec4(RGBAColor(84, 255, 255, 255));
		case '=': return ImVec4(RGBAColor(205, 205, 105, 255));
		case '/': return ImVec4(RGBAColor(146, 146, 252, 255));
	}

	return ImVec4(RGBAColor(230, 230, 230, 255));
}


const char* itemsJitter[] = { "1", "4", "9" };
const char* itemsJitterMU[] = { "0", "1", "2", "3", "4", "5", "6" };
const char* itemsCudaRays[] = { "8000", "16000", "32000", "64000", "128000", "512000", "1024000"};
const char* lightmap_resolution[] = { "1024", "2048", "4096", "8192", "16384" };

void DrawLCConfig()
{
	ImGui::Checkbox("Lighting Compiler", &gCompilerMode.LC);
	ImGui::Separator();

	{
		ImGui::BeginDisabled(!gCompilerMode.LC);
		
		// Lighting Setup
		ImGui::Spacing();
		ImGui::TextColored(getLogColor_new("$"), "Lighting Setup: ");
 
		ImGui::Checkbox("DXT1 Availd Lighting", &gCompilerMode.LC_Dxt1Avail);
 		ImGui::Checkbox("No Static Map",		&gCompilerMode.LC_NoRGB);
	
		// Geometry Setup
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::TextColored(getLogColor_new("!"), "Geometry Setup: ");
 
		ImGui::Checkbox("No Smooth Group", &gCompilerMode.LC_NoSMG);
		ImGui::Checkbox("Remove invalid faces", &gCompilerMode.LC_RemoveInvalidFaces);
		ImGui::Checkbox("Skip invalid faces", &gCompilerMode.LC_SkipInvalidFaces);
		ImGui::Checkbox("Skip Welding", &gCompilerMode.LC_skipWeld);
		ImGui::Checkbox("Tesselation", &gCompilerMode.LC_Tess);

		// Geometry Optimizing After Build
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::TextColored(getLogColor_new("!"), "Geometry OGF Optimize: ");
 
		ImGui::Checkbox("Make TangentBasis", &gCompilerMode.LC_Tangent);
		ImGui::Checkbox("Make Progressive",  &gCompilerMode.LC_MakeProgressive);
		ImGui::Checkbox("Make Striptify",    &gCompilerMode.LC_MakeStriptify);


		// Lightmaps Setup
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::TextColored(getLogColor_new("*"), "Lightmaps Setup: ");
 
		ImGui::Text("BORDER:"); ImGui::SameLine(0, 30);
		ImGui::InputInt("##border", &gCompilerMode.LC_lmap_BORDER, 1, 1);

		ImGui::Text("Size:  "); ImGui::SameLine(0, 30);
		ImGui::Combo("##lmaps", &gCompilerMode.item_lmap_selected, lightmap_resolution, 5);
		gCompilerMode.LC_lmap_size = atoi(lightmap_resolution[gCompilerMode.item_lmap_selected]);

		ImGui::Text("Fill:  "); ImGui::SameLine(0, 30);
		ImGui::InputFloat("##fill", &gCompilerMode.LC_lmap_fill, 0.01f, 0.01f);
		ImGui::Checkbox("Fast lmaps", &gCompilerMode.LC_Se7kills_method);

		ImGui::EndDisabled();
	}

}

void DrawDOConfig()
{
	//if (ImGui::BeginChild("DO", { 200, 370 }, ImGuiChildFlags_Border, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings))
	{
		ImGui::Checkbox("Details Compiler", &gCompilerMode.DO);
		ImGui::Separator();

		ImGui::BeginDisabled(!gCompilerMode.DO);
		ImGui::Checkbox("No Sun", &gCompilerMode.LC_NoSun);
 		ImGui::EndDisabled();
 	}

}

void DrawAIConfig()
{
	//if (ImGui::BeginChild("AI", { 200, 370 }, ImGuiChildFlags_Border, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings))
	{
		ImGui::Checkbox("AI Compiler", &gCompilerMode.AI);

		ImGui::BeginDisabled(!gCompilerMode.AI);

		ImGui::Separator();

		ImGui::Checkbox("AI Compiler ai.level", &gCompilerMode.AI_BuildLevel);

		ImGui::BeginDisabled(!gCompilerMode.AI_BuildLevel);

		ImGui::Checkbox("AI-Map V11", &gCompilerMode.AI_Map_NoLimits);
		ImGui::Checkbox("Draft AI-Map", &gCompilerMode.AI_Draft);
		ImGui::Checkbox("Pure Covers", &gCompilerMode.AI_PureCovers);
		ImGui::Checkbox("Verify", &gCompilerMode.AI_Verify);
		ImGui::Checkbox("Verbose", &gCompilerMode.AI_Verbose);

		ImGui::EndDisabled();

		ImGui::Separator();

		ImGui::Checkbox("AI Compiler all.spawn", &gCompilerMode.AI_BuildSpawn);
		ImGui::BeginDisabled(!gCompilerMode.AI_BuildSpawn);

		ImGui::Checkbox("No Separator Check", &gCompilerMode.AI_NoSeparatorCheck);

		ImGui::Text("Name all.spawn :");
		ImGui::InputText("#1", gCompilerMode.AI_spawn_name, sizeof(gCompilerMode.AI_spawn_name));
		ImGui::Text("Name level start:");
		ImGui::InputText("#2", gCompilerMode.AI_StartActor, sizeof(gCompilerMode.AI_StartActor));

		ImGui::EndDisabled();

		ImGui::EndDisabled();
 	}
}

void DrawCompilerConfig()
{
	ImGui::Checkbox("Silent mode", &gCompilerMode.Silent);

	ImGui::PushID("LightPreset");

	{
  		ImGui::RadioButton("Use Intel Embree", &gCompilerMode.RadioID, 0);
 		ImGui::RadioButton("Use Nvidia CUDA",  &gCompilerMode.RadioID, 1);
 
		switch (gCompilerMode.RadioID)
		{
 			case 0: gCompilerMode.CUDA = false; gCompilerMode.Embree = true; break;
			case 1: gCompilerMode.CUDA = true;  gCompilerMode.Embree = false; break;
			default: break;
		}
	}
	ImGui::PopID();
	ImGui::Separator();

	ImGui::BeginDisabled(!gCompilerMode.Embree);
	ImGui::TextColored(ImVec4(RGBAColor(0, 255, 0, 255)), "(This Only For Build BVH)");
	ImGui::Checkbox("Embree Compacted", &gCompilerMode.EmbreeBVHCompact);
	ImGui::Checkbox("Embree Robust", &gCompilerMode.EmbreeBVHRobust);
	ImGui::Checkbox("Embree AVX2 mode", &gCompilerMode.use_avx2);

	ImGui::EndDisabled();

	if (true)
	{
		ImGui::BeginDisabled(!gCompilerMode.CUDA);

		ImGui::Combo("CudaRays", &gCompilerMode.item_cuda_rays, itemsCudaRays, 7);
		gCompilerMode.LC_CUDA_RAYS_SIZE = atoi(itemsCudaRays[gCompilerMode.item_cuda_rays]);

		ImGui::EndDisabled();
	}


	ImGui::Separator();

	ImGui::SetNextItemWidth(100);
	ImGui::InputInt("Threads", &gCompilerMode.ThreadsNum);
 
	ImGui::Separator();
	
	if (true)
	{
		ImGui::Checkbox("Overload Prebuild", &gCompilerMode.IsOverloadedSettings);

		ImGui::BeginDisabled(!gCompilerMode.IsOverloadedSettings);
		ImGui::SetNextItemWidth(100);
		ImGui::Combo("JitterMU", &gCompilerMode.item_current_jitter_mu, itemsJitterMU, 7);
		ImGui::SetNextItemWidth(100);
		ImGui::Combo("Jitter", &gCompilerMode.item_current_jitter, itemsJitter, 3);
		ImGui::SetNextItemWidth(100);
		ImGui::InputFloat("Pixels", &gCompilerMode.LC_Pixels);
		ImGui::SetNextItemWidth(100);
		ImGui::InputFloat("Dist Weld", &gCompilerMode.LC_WeldDistance);

		gCompilerMode.LC_JSample   = atoi(itemsJitter  [gCompilerMode.item_current_jitter]);
		gCompilerMode.LC_JSampleMU = atoi(itemsJitterMU[gCompilerMode.item_current_jitter_mu]);

		ImGui::EndDisabled();
	}
	
}


// Update State Console

void getStatusInfo(IterationStatus status, xr_string& text, ImVec4& textCol, char& icon)
{
	switch (status)
	{
	case Complited:
		text = "Complited";
		textCol = { 0, 0.9, 0, 1 };

		icon = 'C';
		break;
	case InProgress:
		text = "In Progress";
		textCol = { 0.9, 0.9, 0, 1 };

		icon = 'B';
		break;
	case Pending:
		text = "Pending";
		textCol = { 0.8, 0.8, 0.8, 0.8 };

		icon = 'A';
		break;
	case Skip:
		text = "Skip";
		textCol = { 0.9, 0.9, 0.9, 0.6 };

		icon = 'D';
		break;
	default:
		text = "";
		textCol = { 1,1,1,1 };

		icon = 'A';
		break;
	}
}

const ImVec4 getLogColor(const char& c)
{
	switch (c)
	{
	case '~': return ImVec4(RGBAColor(248, 248, 49, 255));
	case '!': return ImVec4(RGBAColor(204, 102, 102, 255));
	case '@': return ImVec4(RGBAColor(125, 125, 241, 255));
	case '#': return ImVec4(RGBAColor(0, 222, 205, 155));
	case '$': return ImVec4(RGBAColor(172, 172, 255, 255));
	case '%': return ImVec4(RGBAColor(202, 85, 219, 155));
	case '^': return ImVec4(RGBAColor(100, 246, 121, 255));
	case '&': return ImVec4(RGBAColor(255, 255, 0, 255));
	case '*': return ImVec4(RGBAColor(187, 187, 187, 255));
	case '-': return ImVec4(RGBAColor(0, 255, 0, 255));
	case '+': return ImVec4(RGBAColor(84, 255, 255, 255));
	case '=': return ImVec4(RGBAColor(205, 205, 105, 255));
	case '/': return ImVec4(RGBAColor(146, 146, 252, 255));
	default: return ImVec4(RGBAColor(230, 230, 230, 255));
	}
}
 
// call this with your NVML buffer
void DrawPurpleGpuGraph(const float* values, int count, float maxValue = 100.0f)
{
 	ImVec2 size = ImVec2(520, 110);
	ImVec2 p = ImGui::GetCursorScreenPos();
	ImDrawList* draw = ImGui::GetWindowDrawList();

	// background (dark)
	draw->AddRectFilled(
		p,
		ImVec2(p.x + size.x, p.y + size.y),
		IM_COL32(18, 12, 30, 255)
	);

	// grid
	for (int i = 0; i <= 4; i++)
	{
		float y = p.y + (size.y / 4.0f) * i;
		draw->AddLine(
			ImVec2(p.x, y),
			ImVec2(p.x + size.x, y),
			IM_COL32(80, 40, 120, 80)
		);
	}

	if (count > 1)
	{
		float step = size.x / (float)(count - 1);

		// ---- FILL (purple glow under graph)
		for (int i = 1; i < count; i++)
		{
			float v0 = values[i - 1] / maxValue;
			float v1 = values[i] / maxValue;

			ImVec2 a = ImVec2(p.x + step * (i - 1), p.y + size.y);
			ImVec2 b = ImVec2(p.x + step * (i - 1), p.y + size.y - v0 * size.y);
			ImVec2 c = ImVec2(p.x + step * i, p.y + size.y - v1 * size.y);
			ImVec2 d = ImVec2(p.x + step * i, p.y + size.y);

			draw->AddQuadFilled(
				a, b, c, d,
				IM_COL32(140, 60, 220, 40)
			);
		}

		// ---- GLOW LINE (outer)
		for (int i = 1; i < count; i++)
		{
			float v0 = values[i - 1] / maxValue;
			float v1 = values[i] / maxValue;

			ImVec2 a = ImVec2(
				p.x + step * (i - 1),
				p.y + size.y - v0 * size.y
			);

			ImVec2 b = ImVec2(
				p.x + step * i,
				p.y + size.y - v1 * size.y
			);

			draw->AddLine(
				a, b,
				IM_COL32(180, 80, 255, 60),
				4.0f
			);
		}

		// ---- CORE LINE (sharp purple)
		for (int i = 1; i < count; i++)
		{
			float v0 = values[i - 1] / maxValue;
			float v1 = values[i] / maxValue;

			ImVec2 a = ImVec2(
				p.x + step * (i - 1),
				p.y + size.y - v0 * size.y
			);

			ImVec2 b = ImVec2(
				p.x + step * i,
				p.y + size.y - v1 * size.y
			);

			draw->AddLine(
				a, b,
				IM_COL32(200, 120, 255, 255),
				2.0f
			);
		}
	}

	ImGui::Dummy(size);
}

void DrawGpuGraph(const float* values, int count, float maxValue = 100.0f)
{
	if (count == 0)return;

// 	ImGui::Begin("GPU Monitor");
 
	ImVec2 size = ImVec2(500, 100);
	ImVec2 p = ImGui::GetCursorScreenPos();
	ImDrawList* draw = ImGui::GetWindowDrawList();

	// background
	draw->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), IM_COL32(20, 20, 20, 255));
	
	// grid
	for (int i = 0; i < 5; i++)
	{
		float y = p.y + (size.y / 4) * i;
		draw->AddLine(ImVec2(p.x, y), ImVec2(p.x + size.x, y), IM_COL32(50, 50, 50, 120));
 	}

	// graph line
	float step = size.x / (float)(count - 1);

	for (int i = 1; i < count; i++)
	{
		float v0 = values[i - 1] / maxValue;
		float v1 = values[i] / maxValue;

		ImVec2 a = ImVec2(
			p.x + step * (i - 1),
			p.y + size.y - (v0 * size.y)
		);

		ImVec2 b = ImVec2(
			p.x + step * i,
			p.y + size.y - (v1 * size.y)
		);

		draw->AddLine(a, b, IM_COL32(0, 200, 255, 255), 2.0f);
	}

	// fill (like MSI Afterburner)
	for (int i = 1; i < count; i++)
	{
		float v0 = values[i - 1] / maxValue;
		float v1 = values[i] / maxValue;

		ImVec2 a = ImVec2(p.x + step * (i - 1), p.y + size.y);
		ImVec2 b = ImVec2(p.x + step * (i - 1), p.y + size.y - (v0 * size.y));
		ImVec2 c = ImVec2(p.x + step * i, p.y + size.y - (v1 * size.y));
		ImVec2 d = ImVec2(p.x + step * i, p.y + size.y);

		draw->AddQuadFilled(a, b, c, d, IM_COL32(0, 120, 255, 40));
	}

	ImGui::Dummy(size);
//	ImGui::End();
}

void RenderCompilerUI(int X, int Y)
{
	static bool hideLogSection = false;
	static bool ResizeMaximal = true;
 	// Set up the window
	if (ImGui::Begin("Compile Split Screen", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNavFocus))
	{
		// Calculate sizes for the top and bottom parts
		ImVec2 windowSize = ImGui::GetContentRegionAvail();
		float topHeight = hideLogSection ? windowSize.y - 58.f : windowSize.y * 0.5f;

		// Top section
		if (ImGui::BeginChild("TopSection", ImVec2(windowSize.x, topHeight), true))
		{
			// Level name
			xr_string Levels;

			for (auto& [Name, Selected] : gCompilerMode.Files)
			{
				if (Selected)
					Levels += (!Levels.empty() ? ", " : "") + Name;
			}
			ImGui::Text("%s", Levels.c_str());
			ImGui::Separator();

			ImVec4 phaseTextCol = { 78, 178, 98, 0.78 };
			if (X != 1600 || Y != 900)
			 	SDL_SetWindowSize(g_AppInfo.Window, 1600, 900);

 			// Table
			if (ImGui::BeginTable("IterationsTable", 9, ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
				
				ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_WidthFixed, 15.0f);
				ImGui::TableSetupColumn("Task", ImGuiTableColumnFlags_WidthFixed, 15.f);
				ImGui::TableSetupColumn("Phase", ImGuiTableColumnFlags_WidthFixed, 350.f);
				ImGui::TableSetupColumn("Phase %", ImGuiTableColumnFlags_WidthFixed, 50.f);
				ImGui::TableSetupColumn("Elapsed Time", ImGuiTableColumnFlags_WidthFixed, 100.0f);
				ImGui::TableSetupColumn("Remain Time", ImGuiTableColumnFlags_WidthFixed, 100.0f);
 				ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 60.f);
				ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  				ImGui::TableSetupColumn("Information", ImGuiTableColumnFlags_WidthFixed, 350.0f);

				ImGui::TableHeadersRow();


				for (auto& row : GetIterationData()) {

					xr_string rowStatus;
					ImVec4 rowStatusColor;

					char rowIcon;

					getStatusInfo(row.status, rowStatus, rowStatusColor, rowIcon);

					ImGui::TableNextRow();

					// Status icon

					ImGui::TableSetColumnIndex(0);
					ImGui::PushFont(gCompilerMode.CompilerIconsFont);
					ImGui::TextColored(rowStatusColor, "%c", rowIcon);
					ImGui::PopFont();

					// TASK
					ImGui::TableSetColumnIndex(1);
					ImGui::Text("%s", row.iterationName.c_str());

					// 
					ImGui::TableSetColumnIndex(3);
					ImGui::Text("%0.f", row.Persent * 100);

					// ImGui::TableSetColumnIndex(6);
					// ImGui::Text("%d", row.warnings);
					
					// Status text
					ImGui::TableSetColumnIndex(6);
					ImGui::TextColored(rowStatusColor, rowStatus.c_str());

					for (auto& phase : row.phases)
					{
						xr_string status;
						ImVec4 statusColor;
						char phaseIcon;

						getStatusInfo(phase.status, status, statusColor, phaseIcon);

						ImGui::TableNextRow();

						ImGui::TableSetColumnIndex(1);
						ImGui::PushFont(gCompilerMode.CompilerIconsFont);

						float column_width = ImGui::GetColumnWidth();
						float text_size = ImGui::CalcTextSize("A").x;
						ImGui::SetCursorPosX(ImGui::GetCursorPosX() + column_width - text_size);

						ImGui::TextColored(statusColor, "%c", phaseIcon);

						ImGui::PopFont();

						ImGui::TableSetColumnIndex(2);
						if (phase.PhaseName.size() > 0)
							ImGui::TextColored(phaseTextCol, phase.PhaseName.c_str());
						else 
							ImGui::TextColored(phaseTextCol, "phase is brocken");
						//PHASE %
						auto pers = phase.PhasePersent;

						if (phase.status != Complited) {
							u32 dwCurrentTime = timeGetTime();
							u32 dwTimeDiff = dwCurrentTime - GetPhaseStartTime();
							u32 secElapsed = dwTimeDiff / 1000;
							u32 secRemain = u32(float(secElapsed) / pers) - secElapsed;

							phase.elapsed_time = secElapsed;
							if (pers > 0.005f)
								phase.remain_time = secRemain;
						}

						//
						if (phase.status == Complited)
							pers = 1;
						else if (pers > 1.f)
							pers = 1;
						else if (pers < 0.f)
							pers = 0;

						ImGui::TableSetColumnIndex(3);
						ImGui::TextColored(phaseTextCol, "%0.f", pers * 100);

						ImGui::TableSetColumnIndex(4);
						ImGui::TextColored(phaseTextCol, "%s", make_time(phase.elapsed_time).c_str());

						ImGui::TableSetColumnIndex(5);
						if (phase.status != Complited)
							ImGui::TextColored(phaseTextCol, "%s", (phase.remain_time == 0 ? "Calculating..." : make_time(phase.remain_time).c_str()));

						ImGui::TableSetColumnIndex(6);

						ImGui::TextColored(statusColor, status.c_str());

						ImGui::TableSetColumnIndex(7);
						ImGui::Text("%u MB", u32(size_t(phase.used_memory / 1024 / 1024)));

 						ImGui::TableSetColumnIndex(8);
						ImGui::Text("%s", phase.AdditionalData.c_str());
 					}
				}

				if (autoScroll)
					ImGui::SetScrollY(ImGui::GetScrollMaxY());
				ImGui::EndTable();
			}

			ImGui::EndChild();

			ImGui::Separator();
		}
		 
		// draw LOG
		if (true) 
		{
			ImGui::Text("Log");
			ImGui::SameLine();

			const char* buttonText = (hideLogSection) ? "+" : "-";
			ImVec2 textSize = ImGui::CalcTextSize(buttonText);
			ImVec2 buttonSize = ImVec2(textSize.x + ImGui::GetStyle().FramePadding.x * 2, textSize.y + ImGui::GetStyle().FramePadding.y * 2);

			auto ZSize = ImGui::GetContentRegionAvail();

			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ZSize.x - buttonSize.x);

			if (ImGui::Button(buttonText))
				hideLogSection = !hideLogSection;

			if (!hideLogSection)
			{
 				if (ImGui::BeginChild("LogSection", ImVec2(windowSize.x, windowSize.y - topHeight - (buttonSize.y * 2) - 30), true))
				{
					extern void CudaUsage(unsigned int& UsageCuda, unsigned int& UsageMemory);
					extern  void CudaStatisticThread();
					extern	xr_vector<float> get_cuda_usage();
					extern  xr_vector<float> get_mem_usage();

 					static bool isGpuStarted = false;
 					if (!isGpuStarted)
					{
						isGpuStarted = true;
						CudaStatisticThread();
					}
					
					int Size = windowSize.x / 4;
					if (ImGui::BeginChild("LogWindow", ImVec2(Size*3, 0)))
					{
						ImGuiListClipper clipper;
						clipper.Begin(GetLogVector().size());
						while (clipper.Step())
						{
							for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
							{
								auto& line = GetLogVector()[i];
								ImGui::TextColored(getLogColor_new((char*)line.c_str()), "%s", line.c_str());
							}
						}

 						if (autoScroll)
							ImGui::SetScrollY(ImGui::GetScrollMaxY());

						ImGui::EndChild();
					}

					ImGui::SameLine();

					if (ImGui::BeginChild("GpuUsage", ImVec2(Size, 0)))
					{
 						unsigned int UsageCuda = 0, UsageMemory = 0;
						CudaUsage(UsageCuda, UsageMemory);

						ImGui::Text("Gpu Usage: %u", UsageCuda);
						auto& data = get_cuda_usage();
   						DrawGpuGraph(data.data(), data.size(), 100.0f);

						ImGui::Separator();


						ImGui::Text("Gpu Memory Usage: %u", UsageMemory);
						auto& data_mem = get_mem_usage();
						DrawGpuGraph(data_mem.data(), data_mem.size(), 100.0f);
  						ImGui::EndChild();
					}

					ImGui::EndChild();
				}
			}

			ImGui::Separator();
		}
		 
	
		// draw bottom buttons
 		DrawBottonUI();		
		ImGui::End();
	}	
}

