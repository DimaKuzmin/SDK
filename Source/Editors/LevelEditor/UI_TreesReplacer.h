#pragma once
class UI_TreesReplacer   :  public XrUI
{
static UI_TreesReplacer* Form_Tree;	

private: 
	virtual void Draw() override;
	void FunctionSave();
	void FunctionSave_OBJECTS();
public:
	static void Update();
	bool Refresh_ObjectsRef();
	bool ReadSceneObjects_RefUsed();
	void ReplaceLTX();
	static void Show();
	static void Close();

	string_path search_prefix = "";
	string_path search_prefix_o = "";
 
	ImTextureID m_RealTexture_scene;
	ImTextureID m_RealTexture_replace;	 

	int last_selected = 0;
    int last_selected_object = 0;
	
	int current_item = 0;
	int current_item_object = 0;

	xr_vector<shared_str> refs_vec;
	xr_vector<shared_str> refs_objects_vec;

	bool ShowWindowsList = false;
	
	static IC bool IsOpen()  { return Form_Tree; }
};

