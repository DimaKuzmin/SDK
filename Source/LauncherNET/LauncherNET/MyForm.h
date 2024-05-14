#pragma once

namespace LauncherNET 
{

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
 
	/// <summary>
	/// Сводка для MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: добавьте код конструктора
			//
		}

	protected:
		/// <summary>
		/// Освободить все используемые ресурсы.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TabControl^ TabControl;
	protected:
	private: System::Windows::Forms::TabPage^ Status_Tab;
	private: System::Windows::Forms::TabPage^ Geometry_Tab;
	private: System::Windows::Forms::TabPage^ AI_Tab;
	private: System::Windows::Forms::TabPage^ xrDO;
	private: System::Windows::Forms::TabPage^ TODO;
	
	

	public: System::Windows::Forms::ListBox^ listBox1;


	public: System::Windows::Forms::CheckedListBox^ FlagsCompiler;


	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::TextBox^ ThreadsCount;




	private: System::Windows::Forms::TextBox^ MUSamples;

	private: System::Windows::Forms::TextBox^ Samples;
	private: System::Windows::Forms::TextBox^ PXPM;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::Label^ label5;
	private: System::Windows::Forms::TextBox^ LevelName;
	private: System::Windows::Forms::CheckBox^ UpdatingListBox;
	private: System::Windows::Forms::ListBox^ InfoPhases;
	private: System::Windows::Forms::Label^ label9;
	private: System::Windows::Forms::Label^ label8;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::GroupBox^ IntelEmbreType;
	private: System::Windows::Forms::RadioButton^ RadioEmbreeGUltra;



	private: System::Windows::Forms::RadioButton^ RadioEmbreeGHigh;

	private: System::Windows::Forms::RadioButton^ RadioEmbreeGMedium;

	private: System::Windows::Forms::RadioButton^ RadioEmbreeGLow;
	private: System::Windows::Forms::RadioButton^ RadioEmbreeG_Robust;


	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::RadioButton^ RadioEmbreeGdefault;
	private: System::Windows::Forms::Label^ InfoStatus;
	private: System::Windows::Forms::Label^ BuildTime;
	private: System::Windows::Forms::TextBox^ EmbreeTnear;
	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::CheckBox^ off_mulight;


	private: System::Windows::Forms::CheckBox^ off_lmaps;

	private: System::Windows::Forms::CheckBox^ off_implicit;
	private: System::Windows::Forms::CheckBox^ useDXT1;
	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::TextBox^ EmbreeHitsCollect;
	private: System::Windows::Forms::GroupBox^ groupBox4;
	private: System::Windows::Forms::Label^ label14;
	private: System::Windows::Forms::Label^ label13;
	private: System::Windows::Forms::Label^ label12;
	private: System::Windows::Forms::RichTextBox^ xrAI_LevelsName_Spawn;
	private: System::Windows::Forms::RichTextBox^ xrAI_SpawnOut;
	private: System::Windows::Forms::RichTextBox^ xrAI_SPStartLevel;





	private: System::Windows::Forms::GroupBox^ groupBox3;
	private: System::Windows::Forms::TextBox^ xrAI_LevelName;
	private: System::Windows::Forms::CheckBox^ xrAI_PureCovers;



private: System::Windows::Forms::CheckBox^ xrAI_Draft;

	private: System::Windows::Forms::Label^ label15;
	private: System::Windows::Forms::CheckBox^ xrAI_NoSepartor;
private: System::Windows::Forms::Button^ xrAI_StartSpawn;
private: System::Windows::Forms::Button^ xrAI_SpawnAIMap;
private: System::Windows::Forms::CheckBox^ xrAI_Verify;
private: System::Windows::Forms::Label^ label16;
private: System::Windows::Forms::CheckBox^ xrLC_MUModelsRegresion;
private: System::Windows::Forms::Label^ label17;
private: System::Windows::Forms::Label^ label18;










	private:
		/// <summary>
		/// Обязательная переменная конструктора.
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Требуемый метод для поддержки конструктора — не изменяйте 
		/// содержимое этого метода с помощью редактора кода.
		/// </summary>
		void InitializeComponent(void)
		{
			this->TabControl = (gcnew System::Windows::Forms::TabControl());
			this->Status_Tab = (gcnew System::Windows::Forms::TabPage());
			this->BuildTime = (gcnew System::Windows::Forms::Label());
			this->InfoStatus = (gcnew System::Windows::Forms::Label());
			this->InfoPhases = (gcnew System::Windows::Forms::ListBox());
			this->UpdatingListBox = (gcnew System::Windows::Forms::CheckBox());
			this->listBox1 = (gcnew System::Windows::Forms::ListBox());
			this->Geometry_Tab = (gcnew System::Windows::Forms::TabPage());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->xrLC_MUModelsRegresion = (gcnew System::Windows::Forms::CheckBox());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->EmbreeHitsCollect = (gcnew System::Windows::Forms::TextBox());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->useDXT1 = (gcnew System::Windows::Forms::CheckBox());
			this->off_mulight = (gcnew System::Windows::Forms::CheckBox());
			this->off_lmaps = (gcnew System::Windows::Forms::CheckBox());
			this->off_implicit = (gcnew System::Windows::Forms::CheckBox());
			this->EmbreeTnear = (gcnew System::Windows::Forms::TextBox());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->RadioEmbreeGdefault = (gcnew System::Windows::Forms::RadioButton());
			this->RadioEmbreeG_Robust = (gcnew System::Windows::Forms::RadioButton());
			this->IntelEmbreType = (gcnew System::Windows::Forms::GroupBox());
			this->RadioEmbreeGUltra = (gcnew System::Windows::Forms::RadioButton());
			this->RadioEmbreeGHigh = (gcnew System::Windows::Forms::RadioButton());
			this->RadioEmbreeGMedium = (gcnew System::Windows::Forms::RadioButton());
			this->RadioEmbreeGLow = (gcnew System::Windows::Forms::RadioButton());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->LevelName = (gcnew System::Windows::Forms::TextBox());
			this->PXPM = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->MUSamples = (gcnew System::Windows::Forms::TextBox());
			this->Samples = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->ThreadsCount = (gcnew System::Windows::Forms::TextBox());
			this->FlagsCompiler = (gcnew System::Windows::Forms::CheckedListBox());
			this->AI_Tab = (gcnew System::Windows::Forms::TabPage());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			this->xrAI_NoSepartor = (gcnew System::Windows::Forms::CheckBox());
			this->xrAI_StartSpawn = (gcnew System::Windows::Forms::Button());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->xrAI_LevelsName_Spawn = (gcnew System::Windows::Forms::RichTextBox());
			this->xrAI_SpawnOut = (gcnew System::Windows::Forms::RichTextBox());
			this->xrAI_SPStartLevel = (gcnew System::Windows::Forms::RichTextBox());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->xrAI_Verify = (gcnew System::Windows::Forms::CheckBox());
			this->xrAI_SpawnAIMap = (gcnew System::Windows::Forms::Button());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->xrAI_LevelName = (gcnew System::Windows::Forms::TextBox());
			this->xrAI_PureCovers = (gcnew System::Windows::Forms::CheckBox());
			this->xrAI_Draft = (gcnew System::Windows::Forms::CheckBox());
			this->xrDO = (gcnew System::Windows::Forms::TabPage());
			this->TODO = (gcnew System::Windows::Forms::TabPage());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->TabControl->SuspendLayout();
			this->Status_Tab->SuspendLayout();
			this->Geometry_Tab->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->IntelEmbreType->SuspendLayout();
			this->AI_Tab->SuspendLayout();
			this->groupBox4->SuspendLayout();
			this->groupBox3->SuspendLayout();
			this->SuspendLayout();
			// 
			// TabControl
			// 
			this->TabControl->Controls->Add(this->Status_Tab);
			this->TabControl->Controls->Add(this->Geometry_Tab);
			this->TabControl->Controls->Add(this->AI_Tab);
			this->TabControl->Controls->Add(this->xrDO);
			this->TabControl->Controls->Add(this->TODO);
			this->TabControl->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->TabControl->Location = System::Drawing::Point(12, 12);
			this->TabControl->Name = L"TabControl";
			this->TabControl->SelectedIndex = 0;
			this->TabControl->Size = System::Drawing::Size(1447, 809);
			this->TabControl->TabIndex = 3;
			// 
			// Status_Tab
			// 
			this->Status_Tab->BackColor = System::Drawing::Color::DimGray;
			this->Status_Tab->Controls->Add(this->BuildTime);
			this->Status_Tab->Controls->Add(this->InfoStatus);
			this->Status_Tab->Controls->Add(this->InfoPhases);
			this->Status_Tab->Controls->Add(this->UpdatingListBox);
			this->Status_Tab->Controls->Add(this->listBox1);
			this->Status_Tab->Font = (gcnew System::Drawing::Font(L"ScriptS_IV25", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->Status_Tab->Location = System::Drawing::Point(4, 36);
			this->Status_Tab->Name = L"Status_Tab";
			this->Status_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Status_Tab->Size = System::Drawing::Size(1439, 769);
			this->Status_Tab->TabIndex = 0;
			this->Status_Tab->Text = L"Состояние компиляции";
			// 
			// BuildTime
			// 
			this->BuildTime->BackColor = System::Drawing::Color::Black;
			this->BuildTime->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->BuildTime->FlatStyle = System::Windows::Forms::FlatStyle::System;
			this->BuildTime->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 24, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(204)));
			this->BuildTime->ForeColor = System::Drawing::Color::RosyBrown;
			this->BuildTime->Location = System::Drawing::Point(718, 692);
			this->BuildTime->Name = L"BuildTime";
			this->BuildTime->Size = System::Drawing::Size(284, 77);
			this->BuildTime->TabIndex = 17;
			this->BuildTime->Text = L"I AM TIMER";
			this->BuildTime->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// InfoStatus
			// 
			this->InfoStatus->BackColor = System::Drawing::Color::Black;
			this->InfoStatus->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->InfoStatus->FlatStyle = System::Windows::Forms::FlatStyle::System;
			this->InfoStatus->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(204)));
			this->InfoStatus->ForeColor = System::Drawing::Color::RosyBrown;
			this->InfoStatus->Location = System::Drawing::Point(6, 692);
			this->InfoStatus->Name = L"InfoStatus";
			this->InfoStatus->Size = System::Drawing::Size(706, 85);
			this->InfoStatus->TabIndex = 16;
			// 
			// InfoPhases
			// 
			this->InfoPhases->BackColor = System::Drawing::SystemColors::InfoText;
			this->InfoPhases->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->InfoPhases->ForeColor = System::Drawing::Color::Khaki;
			this->InfoPhases->FormattingEnabled = true;
			this->InfoPhases->ItemHeight = 23;
			this->InfoPhases->Location = System::Drawing::Point(1008, 17);
			this->InfoPhases->Name = L"InfoPhases";
			this->InfoPhases->Size = System::Drawing::Size(413, 740);
			this->InfoPhases->TabIndex = 15;
			// 
			// UpdatingListBox
			// 
			this->UpdatingListBox->AutoSize = true;
			this->UpdatingListBox->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->UpdatingListBox->Location = System::Drawing::Point(6, 658);
			this->UpdatingListBox->Name = L"UpdatingListBox";
			this->UpdatingListBox->Size = System::Drawing::Size(198, 31);
			this->UpdatingListBox->TabIndex = 3;
			this->UpdatingListBox->Text = L"Обновлять список";
			this->UpdatingListBox->UseVisualStyleBackColor = true;
			// 
			// listBox1
			// 
			this->listBox1->BackColor = System::Drawing::SystemColors::ControlText;
			this->listBox1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 12, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->listBox1->ForeColor = System::Drawing::Color::GhostWhite;
			this->listBox1->FormattingEnabled = true;
			this->listBox1->ItemHeight = 23;
			this->listBox1->Location = System::Drawing::Point(6, 17);
			this->listBox1->Name = L"listBox1";
			this->listBox1->Size = System::Drawing::Size(996, 648);
			this->listBox1->TabIndex = 0;
			// 
			// Geometry_Tab
			// 
			this->Geometry_Tab->BackColor = System::Drawing::SystemColors::WindowFrame;
			this->Geometry_Tab->Controls->Add(this->label18);
			this->Geometry_Tab->Controls->Add(this->label17);
			this->Geometry_Tab->Controls->Add(this->label16);
			this->Geometry_Tab->Controls->Add(this->xrLC_MUModelsRegresion);
			this->Geometry_Tab->Controls->Add(this->label11);
			this->Geometry_Tab->Controls->Add(this->EmbreeHitsCollect);
			this->Geometry_Tab->Controls->Add(this->groupBox2);
			this->Geometry_Tab->Controls->Add(this->EmbreeTnear);
			this->Geometry_Tab->Controls->Add(this->label10);
			this->Geometry_Tab->Controls->Add(this->groupBox1);
			this->Geometry_Tab->Controls->Add(this->IntelEmbreType);
			this->Geometry_Tab->Controls->Add(this->button1);
			this->Geometry_Tab->Controls->Add(this->label9);
			this->Geometry_Tab->Controls->Add(this->label8);
			this->Geometry_Tab->Controls->Add(this->label7);
			this->Geometry_Tab->Controls->Add(this->label6);
			this->Geometry_Tab->Controls->Add(this->label5);
			this->Geometry_Tab->Controls->Add(this->LevelName);
			this->Geometry_Tab->Controls->Add(this->PXPM);
			this->Geometry_Tab->Controls->Add(this->label4);
			this->Geometry_Tab->Controls->Add(this->MUSamples);
			this->Geometry_Tab->Controls->Add(this->Samples);
			this->Geometry_Tab->Controls->Add(this->label3);
			this->Geometry_Tab->Controls->Add(this->label2);
			this->Geometry_Tab->Controls->Add(this->label1);
			this->Geometry_Tab->Controls->Add(this->ThreadsCount);
			this->Geometry_Tab->Controls->Add(this->FlagsCompiler);
			this->Geometry_Tab->Location = System::Drawing::Point(4, 36);
			this->Geometry_Tab->Name = L"Geometry_Tab";
			this->Geometry_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Geometry_Tab->Size = System::Drawing::Size(1439, 769);
			this->Geometry_Tab->TabIndex = 1;
			this->Geometry_Tab->Text = L"Настройка Компиляции xrLC";
			// 
			// label17
			// 
			this->label17->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label17->ForeColor = System::Drawing::Color::Firebrick;
			this->label17->Location = System::Drawing::Point(3, 464);
			this->label17->Name = L"label17";
			this->label17->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->label17->Size = System::Drawing::Size(559, 81);
			this->label17->TabIndex = 27;
			this->label17->Text = L"Рекомендовано выберать Intel, для LMAPS, MU Models.\r\nНа Implict стадии плохо проб"
				L"ивает поверхности типа Воды. \r\nВ последующем починится. (Надеюсь).\r\n\r\n\r\n\r\n";
			// 
			// label16
			// 
			this->label16->ForeColor = System::Drawing::Color::Firebrick;
			this->label16->Location = System::Drawing::Point(18, 558);
			this->label16->Name = L"label16";
			this->label16->RightToLeft = System::Windows::Forms::RightToLeft::Yes;
			this->label16->Size = System::Drawing::Size(421, 81);
			this->label16->TabIndex = 26;
			this->label16->Text = L"Эксперементальная фича. Чинит Деревья и затемненые обьекты (Отключением)..\r\n\r\n\r\n";
			// 
			// xrLC_MUModelsRegresion
			// 
			this->xrLC_MUModelsRegresion->AutoSize = true;
			this->xrLC_MUModelsRegresion->Location = System::Drawing::Point(38, 642);
			this->xrLC_MUModelsRegresion->Name = L"xrLC_MUModelsRegresion";
			this->xrLC_MUModelsRegresion->Size = System::Drawing::Size(228, 31);
			this->xrLC_MUModelsRegresion->TabIndex = 4;
			this->xrLC_MUModelsRegresion->Text = L"MU Models Regresion";
			this->xrLC_MUModelsRegresion->UseVisualStyleBackColor = true;
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->ForeColor = System::Drawing::Color::Brown;
			this->label11->Location = System::Drawing::Point(602, 28);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(207, 27);
			this->label11->TabIndex = 25;
			this->label11->Text = L"Embree Hits Per Ray";
			// 
			// EmbreeHitsCollect
			// 
			this->EmbreeHitsCollect->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->EmbreeHitsCollect->Location = System::Drawing::Point(838, 28);
			this->EmbreeHitsCollect->Name = L"EmbreeHitsCollect";
			this->EmbreeHitsCollect->Size = System::Drawing::Size(200, 34);
			this->EmbreeHitsCollect->TabIndex = 24;
			this->EmbreeHitsCollect->Text = L"256";
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->useDXT1);
			this->groupBox2->Controls->Add(this->off_mulight);
			this->groupBox2->Controls->Add(this->off_lmaps);
			this->groupBox2->Controls->Add(this->off_implicit);
			this->groupBox2->Location = System::Drawing::Point(27, 28);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(361, 240);
			this->groupBox2->TabIndex = 23;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Debuging";
			// 
			// useDXT1
			// 
			this->useDXT1->AutoSize = true;
			this->useDXT1->Location = System::Drawing::Point(20, 163);
			this->useDXT1->Name = L"useDXT1";
			this->useDXT1->Size = System::Drawing::Size(219, 31);
			this->useDXT1->TabIndex = 3;
			this->useDXT1->Text = L"use_DXT1 (noAlpha)";
			this->useDXT1->UseVisualStyleBackColor = true;
			// 
			// off_mulight
			// 
			this->off_mulight->AutoSize = true;
			this->off_mulight->Location = System::Drawing::Point(20, 126);
			this->off_mulight->Name = L"off_mulight";
			this->off_mulight->Size = System::Drawing::Size(138, 31);
			this->off_mulight->TabIndex = 2;
			this->off_mulight->Text = L"off_mulight";
			this->off_mulight->UseVisualStyleBackColor = true;
			// 
			// off_lmaps
			// 
			this->off_lmaps->AutoSize = true;
			this->off_lmaps->Location = System::Drawing::Point(20, 89);
			this->off_lmaps->Name = L"off_lmaps";
			this->off_lmaps->Size = System::Drawing::Size(123, 31);
			this->off_lmaps->TabIndex = 1;
			this->off_lmaps->Text = L"off_lmaps";
			this->off_lmaps->UseVisualStyleBackColor = true;
			// 
			// off_implicit
			// 
			this->off_implicit->AutoSize = true;
			this->off_implicit->Location = System::Drawing::Point(20, 52);
			this->off_implicit->Name = L"off_implicit";
			this->off_implicit->Size = System::Drawing::Size(137, 31);
			this->off_implicit->TabIndex = 0;
			this->off_implicit->Text = L"off_implicit";
			this->off_implicit->UseVisualStyleBackColor = true;
			// 
			// EmbreeTnear
			// 
			this->EmbreeTnear->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->EmbreeTnear->Location = System::Drawing::Point(838, 68);
			this->EmbreeTnear->Name = L"EmbreeTnear";
			this->EmbreeTnear->Size = System::Drawing::Size(200, 34);
			this->EmbreeTnear->TabIndex = 22;
			this->EmbreeTnear->Text = L"0.001";
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->ForeColor = System::Drawing::Color::Brown;
			this->label10->Location = System::Drawing::Point(624, 68);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(143, 27);
			this->label10->TabIndex = 21;
			this->label10->Text = L"Embree Tnear";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->RadioEmbreeGdefault);
			this->groupBox1->Controls->Add(this->RadioEmbreeG_Robust);
			this->groupBox1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->groupBox1->Location = System::Drawing::Point(579, 582);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(587, 107);
			this->groupBox1->TabIndex = 20;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Тип Рейтрейсинга";
			// 
			// RadioEmbreeGdefault
			// 
			this->RadioEmbreeGdefault->AutoSize = true;
			this->RadioEmbreeGdefault->Checked = true;
			this->RadioEmbreeGdefault->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->RadioEmbreeGdefault->Location = System::Drawing::Point(10, 70);
			this->RadioEmbreeGdefault->Name = L"RadioEmbreeGdefault";
			this->RadioEmbreeGdefault->Size = System::Drawing::Size(144, 31);
			this->RadioEmbreeGdefault->TabIndex = 5;
			this->RadioEmbreeGdefault->TabStop = true;
			this->RadioEmbreeGdefault->Text = L"DefaultFlags";
			this->RadioEmbreeGdefault->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			this->RadioEmbreeGdefault->UseVisualStyleBackColor = true;
			// 
			// RadioEmbreeG_Robust
			// 
			this->RadioEmbreeG_Robust->AutoSize = true;
			this->RadioEmbreeG_Robust->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->RadioEmbreeG_Robust->Location = System::Drawing::Point(10, 33);
			this->RadioEmbreeG_Robust->Name = L"RadioEmbreeG_Robust";
			this->RadioEmbreeG_Robust->Size = System::Drawing::Size(293, 31);
			this->RadioEmbreeG_Robust->TabIndex = 4;
			this->RadioEmbreeG_Robust->TabStop = true;
			this->RadioEmbreeG_Robust->Text = L"RTC_SCENE_FLAG_ROBUST";
			this->RadioEmbreeG_Robust->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			this->RadioEmbreeG_Robust->UseVisualStyleBackColor = true;
			// 
			// IntelEmbreType
			// 
			this->IntelEmbreType->Controls->Add(this->RadioEmbreeGUltra);
			this->IntelEmbreType->Controls->Add(this->RadioEmbreeGHigh);
			this->IntelEmbreType->Controls->Add(this->RadioEmbreeGMedium);
			this->IntelEmbreType->Controls->Add(this->RadioEmbreeGLow);
			this->IntelEmbreType->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->IntelEmbreType->Location = System::Drawing::Point(575, 440);
			this->IntelEmbreType->Name = L"IntelEmbreType";
			this->IntelEmbreType->Size = System::Drawing::Size(591, 146);
			this->IntelEmbreType->TabIndex = 19;
			this->IntelEmbreType->TabStop = false;
			this->IntelEmbreType->Text = L"Тип Геометрии (Intel)";
			// 
			// RadioEmbreeGUltra
			// 
			this->RadioEmbreeGUltra->AutoSize = true;
			this->RadioEmbreeGUltra->Checked = true;
			this->RadioEmbreeGUltra->Location = System::Drawing::Point(15, 93);
			this->RadioEmbreeGUltra->Name = L"RadioEmbreeGUltra";
			this->RadioEmbreeGUltra->Size = System::Drawing::Size(77, 31);
			this->RadioEmbreeGUltra->TabIndex = 3;
			this->RadioEmbreeGUltra->TabStop = true;
			this->RadioEmbreeGUltra->Text = L"Ultra";
			this->RadioEmbreeGUltra->UseVisualStyleBackColor = true;
			// 
			// RadioEmbreeGHigh
			// 
			this->RadioEmbreeGHigh->AutoSize = true;
			this->RadioEmbreeGHigh->Location = System::Drawing::Point(14, 70);
			this->RadioEmbreeGHigh->Name = L"RadioEmbreeGHigh";
			this->RadioEmbreeGHigh->Size = System::Drawing::Size(71, 31);
			this->RadioEmbreeGHigh->TabIndex = 2;
			this->RadioEmbreeGHigh->TabStop = true;
			this->RadioEmbreeGHigh->Text = L"High";
			this->RadioEmbreeGHigh->UseVisualStyleBackColor = true;
			// 
			// RadioEmbreeGMedium
			// 
			this->RadioEmbreeGMedium->AutoSize = true;
			this->RadioEmbreeGMedium->Location = System::Drawing::Point(14, 47);
			this->RadioEmbreeGMedium->Name = L"RadioEmbreeGMedium";
			this->RadioEmbreeGMedium->Size = System::Drawing::Size(98, 31);
			this->RadioEmbreeGMedium->TabIndex = 1;
			this->RadioEmbreeGMedium->TabStop = true;
			this->RadioEmbreeGMedium->Text = L"Medium";
			this->RadioEmbreeGMedium->UseVisualStyleBackColor = true;
			// 
			// RadioEmbreeGLow
			// 
			this->RadioEmbreeGLow->AutoSize = true;
			this->RadioEmbreeGLow->Location = System::Drawing::Point(15, 22);
			this->RadioEmbreeGLow->Name = L"RadioEmbreeGLow";
			this->RadioEmbreeGLow->Size = System::Drawing::Size(63, 31);
			this->RadioEmbreeGLow->TabIndex = 0;
			this->RadioEmbreeGLow->TabStop = true;
			this->RadioEmbreeGLow->Text = L"Low";
			this->RadioEmbreeGLow->UseVisualStyleBackColor = true;
			// 
			// button1
			// 
			this->button1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(204)));
			this->button1->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)), static_cast<System::Int32>(static_cast<System::Byte>(0)),
				static_cast<System::Int32>(static_cast<System::Byte>(192)));
			this->button1->Location = System::Drawing::Point(579, 707);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(269, 38);
			this->button1->TabIndex = 18;
			this->button1->Text = L"Старт";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click_1);
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label9->ForeColor = System::Drawing::Color::RosyBrown;
			this->label9->Location = System::Drawing::Point(834, 400);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(207, 27);
			this->label9->TabIndex = 17;
			this->label9->Text = L"1-25 (defualt max 10)";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label8->ForeColor = System::Drawing::Color::RosyBrown;
			this->label8->Location = System::Drawing::Point(1058, 335);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(41, 27);
			this->label8->TabIndex = 16;
			this->label8->Text = L"1-6";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label7->ForeColor = System::Drawing::Color::RosyBrown;
			this->label7->Location = System::Drawing::Point(1058, 300);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(61, 27);
			this->label7->TabIndex = 15;
			this->label7->Text = L"1-4-9";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label6->ForeColor = System::Drawing::Color::RosyBrown;
			this->label6->Location = System::Drawing::Point(1058, 265);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(74, 27);
			this->label6->TabIndex = 14;
			this->label6->Text = L"1-MAX";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label5->Location = System::Drawing::Point(622, 213);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(120, 27);
			this->label5->TabIndex = 13;
			this->label5->Text = L"Имя Уровня";
			// 
			// LevelName
			// 
			this->LevelName->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->LevelName->Location = System::Drawing::Point(838, 213);
			this->LevelName->Name = L"LevelName";
			this->LevelName->Size = System::Drawing::Size(200, 34);
			this->LevelName->TabIndex = 12;
			this->LevelName->Text = L"test";
			// 
			// PXPM
			// 
			this->PXPM->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->PXPM->Location = System::Drawing::Point(838, 368);
			this->PXPM->Name = L"PXPM";
			this->PXPM->Size = System::Drawing::Size(203, 29);
			this->PXPM->TabIndex = 11;
			this->PXPM->Text = L"10";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label4->Location = System::Drawing::Point(582, 368);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(250, 27);
			this->label4->TabIndex = 10;
			this->label4->Text = L"Кол-во Пикселей на метр";
			// 
			// MUSamples
			// 
			this->MUSamples->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->MUSamples->Location = System::Drawing::Point(838, 333);
			this->MUSamples->Name = L"MUSamples";
			this->MUSamples->Size = System::Drawing::Size(203, 29);
			this->MUSamples->TabIndex = 9;
			this->MUSamples->Text = L"6";
			// 
			// Samples
			// 
			this->Samples->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->Samples->Location = System::Drawing::Point(838, 298);
			this->Samples->Name = L"Samples";
			this->Samples->Size = System::Drawing::Size(203, 29);
			this->Samples->TabIndex = 8;
			this->Samples->Text = L"9";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label3->Location = System::Drawing::Point(599, 335);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(210, 27);
			this->label3->TabIndex = 4;
			this->label3->Text = L"Кол-во MU-SAMPLES";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label2->Location = System::Drawing::Point(620, 300);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(162, 27);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Кол-во Семплов";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label1->Location = System::Drawing::Point(622, 265);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(160, 27);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Кол-во Потоков";
			// 
			// ThreadsCount
			// 
			this->ThreadsCount->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->ThreadsCount->Location = System::Drawing::Point(838, 263);
			this->ThreadsCount->Name = L"ThreadsCount";
			this->ThreadsCount->Size = System::Drawing::Size(203, 29);
			this->ThreadsCount->TabIndex = 1;
			this->ThreadsCount->Text = L"16";
			// 
			// FlagsCompiler
			// 
			this->FlagsCompiler->BackColor = System::Drawing::Color::Black;
			this->FlagsCompiler->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 15.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->FlagsCompiler->ForeColor = System::Drawing::Color::DarkViolet;
			this->FlagsCompiler->FormattingEnabled = true;
			this->FlagsCompiler->Location = System::Drawing::Point(1172, 28);
			this->FlagsCompiler->Name = L"FlagsCompiler";
			this->FlagsCompiler->Size = System::Drawing::Size(249, 708);
			this->FlagsCompiler->TabIndex = 0;
			// 
			// AI_Tab
			// 
			this->AI_Tab->BackColor = System::Drawing::Color::DimGray;
			this->AI_Tab->Controls->Add(this->groupBox4);
			this->AI_Tab->Controls->Add(this->groupBox3);
			this->AI_Tab->Location = System::Drawing::Point(4, 36);
			this->AI_Tab->Name = L"AI_Tab";
			this->AI_Tab->Size = System::Drawing::Size(1439, 769);
			this->AI_Tab->TabIndex = 2;
			this->AI_Tab->Text = L"Настройка компиляции xrAI";
			// 
			// groupBox4
			// 
			this->groupBox4->Controls->Add(this->xrAI_NoSepartor);
			this->groupBox4->Controls->Add(this->xrAI_StartSpawn);
			this->groupBox4->Controls->Add(this->label14);
			this->groupBox4->Controls->Add(this->label13);
			this->groupBox4->Controls->Add(this->label12);
			this->groupBox4->Controls->Add(this->xrAI_LevelsName_Spawn);
			this->groupBox4->Controls->Add(this->xrAI_SpawnOut);
			this->groupBox4->Controls->Add(this->xrAI_SPStartLevel);
			this->groupBox4->Location = System::Drawing::Point(390, 33);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Size = System::Drawing::Size(589, 276);
			this->groupBox4->TabIndex = 1;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"AI Spawn";
			// 
			// xrAI_NoSepartor
			// 
			this->xrAI_NoSepartor->AutoSize = true;
			this->xrAI_NoSepartor->Location = System::Drawing::Point(25, 162);
			this->xrAI_NoSepartor->Name = L"xrAI_NoSepartor";
			this->xrAI_NoSepartor->Size = System::Drawing::Size(192, 31);
			this->xrAI_NoSepartor->TabIndex = 8;
			this->xrAI_NoSepartor->Text = L"NoSepartorCheck";
			this->xrAI_NoSepartor->UseVisualStyleBackColor = true;
			// 
			// xrAI_StartSpawn
			// 
			this->xrAI_StartSpawn->Location = System::Drawing::Point(15, 210);
			this->xrAI_StartSpawn->Name = L"xrAI_StartSpawn";
			this->xrAI_StartSpawn->Size = System::Drawing::Size(185, 51);
			this->xrAI_StartSpawn->TabIndex = 8;
			this->xrAI_StartSpawn->Text = L"Запустить Спавн";
			this->xrAI_StartSpawn->UseVisualStyleBackColor = true;
			this->xrAI_StartSpawn->Click += gcnew System::EventHandler(this, &MyForm::xrAI_StartSpawn_Click);
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(20, 116);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(135, 27);
			this->label14->TabIndex = 5;
			this->label14->Text = L"Старт Игрока";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(20, 80);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(72, 27);
			this->label13->TabIndex = 4;
			this->label13->Text = L"Выход";
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Location = System::Drawing::Point(20, 40);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(153, 27);
			this->label12->TabIndex = 3;
			this->label12->Text = L"Уровни Спавна";
			// 
			// xrAI_LevelsName_Spawn
			// 
			this->xrAI_LevelsName_Spawn->Location = System::Drawing::Point(244, 33);
			this->xrAI_LevelsName_Spawn->Name = L"xrAI_LevelsName_Spawn";
			this->xrAI_LevelsName_Spawn->Size = System::Drawing::Size(328, 34);
			this->xrAI_LevelsName_Spawn->TabIndex = 2;
			this->xrAI_LevelsName_Spawn->Text = L"";
			// 
			// xrAI_SpawnOut
			// 
			this->xrAI_SpawnOut->Location = System::Drawing::Point(244, 73);
			this->xrAI_SpawnOut->Name = L"xrAI_SpawnOut";
			this->xrAI_SpawnOut->Size = System::Drawing::Size(328, 34);
			this->xrAI_SpawnOut->TabIndex = 1;
			this->xrAI_SpawnOut->Text = L"";
			// 
			// xrAI_SPStartLevel
			// 
			this->xrAI_SPStartLevel->Location = System::Drawing::Point(244, 113);
			this->xrAI_SPStartLevel->Name = L"xrAI_SPStartLevel";
			this->xrAI_SPStartLevel->Size = System::Drawing::Size(328, 34);
			this->xrAI_SPStartLevel->TabIndex = 0;
			this->xrAI_SPStartLevel->Text = L"";
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->xrAI_Verify);
			this->groupBox3->Controls->Add(this->xrAI_SpawnAIMap);
			this->groupBox3->Controls->Add(this->label15);
			this->groupBox3->Controls->Add(this->xrAI_LevelName);
			this->groupBox3->Controls->Add(this->xrAI_PureCovers);
			this->groupBox3->Controls->Add(this->xrAI_Draft);
			this->groupBox3->Location = System::Drawing::Point(13, 33);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(358, 276);
			this->groupBox3->TabIndex = 0;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"AI Map";
			// 
			// xrAI_Verify
			// 
			this->xrAI_Verify->AutoSize = true;
			this->xrAI_Verify->Location = System::Drawing::Point(10, 104);
			this->xrAI_Verify->Name = L"xrAI_Verify";
			this->xrAI_Verify->Size = System::Drawing::Size(90, 31);
			this->xrAI_Verify->TabIndex = 8;
			this->xrAI_Verify->Text = L"Verify";
			this->xrAI_Verify->UseVisualStyleBackColor = true;
			// 
			// xrAI_SpawnAIMap
			// 
			this->xrAI_SpawnAIMap->Location = System::Drawing::Point(13, 210);
			this->xrAI_SpawnAIMap->Name = L"xrAI_SpawnAIMap";
			this->xrAI_SpawnAIMap->Size = System::Drawing::Size(185, 51);
			this->xrAI_SpawnAIMap->TabIndex = 7;
			this->xrAI_SpawnAIMap->Text = L"Запустить";
			this->xrAI_SpawnAIMap->UseVisualStyleBackColor = true;
			this->xrAI_SpawnAIMap->Click += gcnew System::EventHandler(this, &MyForm::xrAI_SpawnAIMap_Click);
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(8, 162);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(86, 27);
			this->label15->TabIndex = 6;
			this->label15->Text = L"Уровень";
			// 
			// xrAI_LevelName
			// 
			this->xrAI_LevelName->Location = System::Drawing::Point(100, 155);
			this->xrAI_LevelName->Name = L"xrAI_LevelName";
			this->xrAI_LevelName->Size = System::Drawing::Size(240, 34);
			this->xrAI_LevelName->TabIndex = 2;
			// 
			// xrAI_PureCovers
			// 
			this->xrAI_PureCovers->AutoSize = true;
			this->xrAI_PureCovers->Location = System::Drawing::Point(10, 67);
			this->xrAI_PureCovers->Name = L"xrAI_PureCovers";
			this->xrAI_PureCovers->Size = System::Drawing::Size(131, 31);
			this->xrAI_PureCovers->TabIndex = 1;
			this->xrAI_PureCovers->Text = L"PureCovers";
			this->xrAI_PureCovers->UseVisualStyleBackColor = true;
			// 
			// xrAI_Draft
			// 
			this->xrAI_Draft->AutoSize = true;
			this->xrAI_Draft->Location = System::Drawing::Point(10, 30);
			this->xrAI_Draft->Name = L"xrAI_Draft";
			this->xrAI_Draft->Size = System::Drawing::Size(84, 31);
			this->xrAI_Draft->TabIndex = 0;
			this->xrAI_Draft->Text = L"Draft";
			this->xrAI_Draft->UseVisualStyleBackColor = true;
			// 
			// xrDO
			// 
			this->xrDO->BackColor = System::Drawing::Color::DimGray;
			this->xrDO->Location = System::Drawing::Point(4, 36);
			this->xrDO->Name = L"xrDO";
			this->xrDO->Size = System::Drawing::Size(1439, 769);
			this->xrDO->TabIndex = 3;
			this->xrDO->Text = L"Настройка Компиляции xrDO";
			// 
			// TODO
			// 
			this->TODO->BackColor = System::Drawing::Color::DimGray;
			this->TODO->Location = System::Drawing::Point(4, 36);
			this->TODO->Name = L"TODO";
			this->TODO->Size = System::Drawing::Size(1439, 769);
			this->TODO->TabIndex = 4;
			this->TODO->Text = L"TODO";
			// 
			// label18
			// 
			this->label18->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label18->ForeColor = System::Drawing::Color::Firebrick;
			this->label18->Location = System::Drawing::Point(3, 383);
			this->label18->Name = L"label18";
			this->label18->RightToLeft = System::Windows::Forms::RightToLeft::No;
			this->label18->Size = System::Drawing::Size(559, 81);
			this->label18->TabIndex = 28;
			this->label18->Text = L"Стандартный RayTrace Тоже ускорен (Отсечены лишние hits)....";
			// 
			// MyForm
			// 
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::None;
			this->BackColor = System::Drawing::SystemColors::ActiveBorder;
			this->BackgroundImageLayout = System::Windows::Forms::ImageLayout::None;
			this->ClientSize = System::Drawing::Size(1471, 821);
			this->Controls->Add(this->TabControl);
			this->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->ForeColor = System::Drawing::Color::Tomato;
			this->Name = L"MyForm";
			this->ShowIcon = false;
			this->Text = L"Продвинутый компилятор Геометрии и Света (Intel)";
			this->TabControl->ResumeLayout(false);
			this->Status_Tab->ResumeLayout(false);
			this->Status_Tab->PerformLayout();
			this->Geometry_Tab->ResumeLayout(false);
			this->Geometry_Tab->PerformLayout();
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->IntelEmbreType->ResumeLayout(false);
			this->IntelEmbreType->PerformLayout();
			this->AI_Tab->ResumeLayout(false);
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			this->ResumeLayout(false);

		}
		

		 public: System::Void AddItemToListBox_form(System::String^ str)
		 {
			 listBox1->Items->Add(str);
		 }

		public: System::Void AddItemToPhases_form(System::String^ str)
		{
			InfoPhases->Items->Add(str);
		}

		public: System::Void UpdateTextStatus_form(System::String^ str)
		{
			InfoStatus->Text = str;
		}

		public: System::Void UpdateTime_form(System::String^ str)
		{
			BuildTime->Text = str;
		}
 
		// Call From Other Threads Safe
		public: System::Void updateLogFormItem(const char* str)
		{
			// Вызываем метод updateLog из .NET кода с использованием P/Invoke
			System::String^ managedString = gcnew System::String(str);
			this->Invoke(gcnew Action<System::String^>(this, &MyForm::AddItemToListBox_form), managedString);
		}

		public: System::Void updatePhaseItem(const char* str)
		{
			// Вызываем метод updateLog из .NET кода с использованием P/Invoke
			System::String^ managedString = gcnew System::String(str);
			this->Invoke(gcnew Action<System::String^>(this, &MyForm::AddItemToPhases_form), managedString);
		}


		public: System::Void updateStatusItem(const char* str)
		{
			// Вызываем метод updateLog из .NET кода с использованием P/Invoke
			System::String^ managedString = gcnew System::String(str);
			this->Invoke(gcnew Action<System::String^>(this, &MyForm::UpdateTextStatus_form), managedString);
		}
			  
		 public: System::Void updateALL()
		 {
			// if (UpdatingListBox->Checked)
			//	 listBox1->SelectedIndex = listBox1->Items->Count - 1;
		 }


		public:  System::Void UpdateTime(const char* str)
		{
			System::String^ managedString = gcnew System::String(str);
			this->Invoke(gcnew Action<System::String^>(this, &MyForm::UpdateTime_form), managedString);
			//BuildTime->Text = managedString;
		}
 
 
		// Buttons

		private: System::Void button1_Click_1(System::Object^ sender, System::EventArgs^ e);

		private: System::Void xrAI_SpawnAIMap_Click(System::Object^ sender, System::EventArgs^ e);
	 
		private: System::Void xrAI_StartSpawn_Click(System::Object^ sender, System::EventArgs^ e);
 
};

}