#pragma once


class UI_CompilerXR : public XrUI
{
	static UI_CompilerXR* Form;	
	static bool reinit;
 

private:	
 
	bool nosun = false, norgb = false, nohemi = false;
	bool no_invalidatefaces = false, no_optimize = false;

	bool no_simplify = false;
	bool use_intel = false;
	bool use_avx = false;
	bool use_sse = false;

	bool hw_light = false;
	bool use_opcode_older = false;

	int pxpm = 10;
	int sample = 9;
	int mu_samples = 6;
	int th = 16;

	bool noise = false;

	bool draft_aimap = false;
	bool no_separator_check = true;

	string_path levels_spawn = "";
	string_path levels_out   = "";
 	 
	// Унаследовано через XrUI
	virtual void Draw() override;

public:
	static void Update();
	static void Show();
	static void Close();

	void ReadLTX(CInifile* file);
	void WriteLTX(CInifile* file);

	void WriteBytes(IWriter* file);
	void ReadBytes(IReader* file);

	static IC bool IsOpen()  { return Form; }


};

