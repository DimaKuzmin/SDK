#include "stdafx.h"
#include "UI_TexturesReplacerSOC.h"

UI_TexturesReplacerSOC* UI_TexturesReplacerSOC::Form = nullptr;

void UI_TexturesReplacerSOC::Draw()
{
	if (!ImGui::Begin("Textures Replcae SOC", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}

	


	ImGui::End();
}

void UI_TexturesReplacerSOC::Update()
{
	if (Form)
	{
		if (!Form->IsClosed())
		{
			Form->Draw();
		}
		else
		{
 			xr_delete(Form);
		}
	}
}

void UI_TexturesReplacerSOC::Show()
{
	if (Form)
	{
		if (!Form->IsClosed())
		{
			Form->Draw();
		}
		else
		{
 			xr_delete(Form);
		}
	}
}

void UI_TexturesReplacerSOC::Close()
{
	xr_delete(Form);
}
