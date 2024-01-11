#pragma once
class UIDxtConverter :  public XrUI
{
	static UIDxtConverter* Form_DXT;	

private: 
	virtual void Draw() override;
public:
	static void Update();
	static void Show();
	static void Close();

	string_path path_dir;
	string_path path_dir_out;
	
	static IC bool IsOpen()  { return Form_DXT; }
};

