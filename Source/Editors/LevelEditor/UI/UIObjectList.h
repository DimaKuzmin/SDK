#pragma once

class CCustomObject;
 


class UIObjectList:public XrUI
{
	xr_vector<shared_str> Errored_objects;
	xr_vector<CCustomObject*> customdata_objects;

	bool MultiplySelect = false;
	bool IgnoreVisual = false;
	bool IgnoreNotVisual = false;
	bool ShowLOAD = false;
	int DistanceObjects = 0;
	bool use_prefix_refname = false;

	bool use_errored = false;
	bool use_distance = false;
	int _sizetext = 64;
public:
	UIObjectList();
	virtual ~UIObjectList();
	virtual void Draw();
	static void Update();
	static void Show();
	static void Close();
	static IC bool IsOpen()  { return Form; }

	void SetScale(Fvector size);

	void ExportSelectObjects();
	void ExportAllObjects();
	void ExportAIMap();

	void ImportObjects(Fvector offset = Fvector(), bool use_path = false, xr_string path = {0});

	void ImportMultiply();
	void LoadFromMultiply();
	
 	void SaveSelectedObjects();
	void SetTerrainOffsetForAI();
	void CopyTempLODforObjects();
 
	bool ExportDir(xr_string& dir);

	bool LoadAiMAP();
	void AddSceneObjectToList();

	void UndoLoad();
	void SelectLoaded();
	void CheckCustomData();
	void UpdateCustomData();

	void GenSpawnCFG(xr_string section, xr_string map, xr_string prefix);

	void MoveObjectsToOffset(xr_string& load);
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


	xr_vector<Fvector3> getAIPOS(LPCSTR file);

	void UpdateUIObjectList();
	void ListBoxForTypes();

	bool CheckNameForType(CCustomObject* object);
	bool CheckForError(CCustomObject* object);

	void FindALL_Duplicate();
	void LoadErrorsGraphs();

	void FindObjectSector(u16 id);

private:
	static UIObjectList* Form;
	xr_vector<CCustomObject*> objects_selected;
	int merge_ai_map_size = 0;

	void ClearCustomData();

private:
	void DrawObjects();
	void DrawObject(CCustomObject* obj,const char*name);
private:
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


};