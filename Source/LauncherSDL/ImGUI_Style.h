#pragma once

class ImGUIStyleCFG
{
public:
	bool isRedTheme = false;
	void BlackTheme();
	void RedTheme();
	void ImGui_InitializeStyle();
};

extern ImGUIStyleCFG CIMStyle;