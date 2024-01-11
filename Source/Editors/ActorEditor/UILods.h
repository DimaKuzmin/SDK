#pragma once

class UILods   :  public XrUI
{
	static UILods* Form_Tree;

	string_path search_prefix_o;
    bool gen_lods = false;
	int last_gen_object = -1;

	SPBItem* PB	;

	bool Quallyty;
	int current_item_object = 0;
	int last_selected_object = 0;

private: 
		
	virtual void Draw() override;

	ImTextureID m_RealTexture_replace;

public:
	xr_vector<shared_str> refs_objects_vec;

	static void Update();

	bool Refresh_ObjectsRef();
	 
	static void Show();
	static void Close();

	static IC bool IsOpen()  { return Form_Tree; }
};

