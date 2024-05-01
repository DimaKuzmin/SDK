#pragma once

class CCustomObject;
 


class UIObjectList:public XrUI
{
private:
	static UIObjectList* Form;

	// AIMAP
	xr_string last_fileaimap;
	bool ai_ignore_stractures = false;
	xr_map<int, Fvector3> merge_offsets;

	// MOVE OFFSETS
	Fvector3 vec_offset = Fvector().set(0, 0, 0);
	Fvector3 vec_box_min = Fvector().set(0, 0, 0);
	Fvector3 vec_box_max = Fvector().set(0, 0, 0);

	// BOX Выборка Обьектов
	bool use_outside_box = false;
	xr_map<CCustomObject*, Fvector> objects_to_move;

 

private:
	// SPAWNES
	bool IgnoreCombatCovers = false;
	string_path logic_sec = "smart_terrain", path_sec = "level_", name_sec = "test_";

	string_path spawnnew_custom_data;

 	bool current_only_customdata = 0;
	bool IgnoreVisual = false;
	bool IgnoreNotVisual = false;
	xr_vector<shared_str> Errored_objects;
	xr_vector<CCustomObject*> customdata_objects;

public:
	// RENAME CFG
	// PREFIXES
	string_path prefix_cfg_section;
	string_path prefix_cfg_prefix;
	string_path prefix_cfg_map;

	bool use_genarate_cfgs = false;

	string_path rename_prefix_name;


	bool MultiplySelect = false;


	// DISTANCE IN OBJECT LIST ITEMS

	int DistanceObjects = 0;
	bool use_prefix_refname = false;

	bool use_errored = false;
	bool use_distance = false;
	int _sizetext = 64;

	UIObjectList();
	virtual ~UIObjectList();
	virtual void Draw();
	static void Update();
	static void Show();
	static void Close();
	static IC bool IsOpen()  { return Form; }

 private:
	 // GLOBAL MENU CHECKBOXES
	 bool ShowEXPORT = false;
 	 bool ShowLOAD = false;
	 bool ShowRenamer = false;



	void DrawObjects();
	void DrawObject(CCustomObject* obj,const char*name);

	ObjClassID m_cur_cls;
	enum EMode
	{
		M_All,
		M_Visible,
		M_Inbvisible
	};
	EMode m_Mode;
	CCustomObject* m_SelectedObject;
	bool serch_mode;
	string_path m_Filter = {0};
	string_path m_Filter_type = { 0 };
	string_path m_Filter_visual = { 0 };

public:
// NEW se7kills

	void HideCombatCovers();
 
	void ExportSelectObjects();
	void ExportInsideBox();
	void RemoveAllInsideBox();
	
	void SetListToMove();

	void ExportAllObjects();
	void ExportAIMap(Fbox* box, LPCSTR name);

	void ImportObjects(Fvector offset = Fvector(), bool use_path = false, xr_string path = {0});

	void ImportMultiply();
	void LoadFromMultiply();
	
 	void SaveSelectedObjects();
	void SetTerrainOffsetForAI();
	void CopyTempLODforObjects();
 
	bool ExportDir(xr_string& dir);

	bool LoadAiMAP();
 
	void UndoLoad();
	void SelectLoaded();
	void CheckCustomData();
	void UpdateCustomData();

	void GenSpawnCFG(xr_string section, xr_string map, xr_string prefix);

	void MoveObjectsToOffset();
	void CheckDuplicateNames();

	void RenameALLObjectsToObject();

	void BboxSelectedObject();
	void POS_ObjectsToLTX();

	void SelectAIMAPFile();

	void ModifyAIMAPFiles(Fvector pos);
	void MergeAIMAP(u32 file);

	void MergeAI_FromINI(CInifile* file);

	void RenameSelectedObjects();
	void SetCustomData(bool autoNumarate, LPCSTR logic_sec, LPCSTR path_name);

	void CreateLogicConfigs();

	void ClearGraphs();

	xr_vector<Fvector3> getAIPOS(LPCSTR file);

	void UpdateDefaultMeny();
	void UpdateUIObjectList();
	void ListBoxForTypes();

	bool CheckNameForType(CCustomObject* object);
	bool CheckForError(CCustomObject* object);

	void FindALL_Duplicate();
	void LoadErrorsGraphs();

	void FindObjectSector(u16 id);

private:
	xr_vector<CCustomObject*> objects_selected;
	int merge_ai_map_size = 0;

	void ClearCustomData();

 
};