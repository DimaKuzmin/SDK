#pragma once

class UI_TexturesReplacerSOC : public XrUI
{
	static UI_TexturesReplacerSOC* Form;

	// Унаследовано через XrUI
	virtual void Draw() override;

public:
	static void Update();
	static void Show();
	static void Close();
	 
	static IC bool IsOpen() { return Form; }
};