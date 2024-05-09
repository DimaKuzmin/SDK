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

	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::CheckedListBox^ FlagsCompiler;


	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::TextBox^ ThreadsCount;

	private: System::Windows::Forms::ListBox^ InfoPhases;

	private: System::Windows::Forms::ProgressBar^ progressBar1;
	private: System::Windows::Forms::TextBox^ MUSamples;

	private: System::Windows::Forms::TextBox^ Samples;
	private: System::Windows::Forms::TextBox^ PXPM;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::ListBox^ LevelsList;


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
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->listBox1 = (gcnew System::Windows::Forms::ListBox());
			this->Geometry_Tab = (gcnew System::Windows::Forms::TabPage());
			this->PXPM = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->MUSamples = (gcnew System::Windows::Forms::TextBox());
			this->Samples = (gcnew System::Windows::Forms::TextBox());
			this->InfoPhases = (gcnew System::Windows::Forms::ListBox());
			this->progressBar1 = (gcnew System::Windows::Forms::ProgressBar());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->ThreadsCount = (gcnew System::Windows::Forms::TextBox());
			this->FlagsCompiler = (gcnew System::Windows::Forms::CheckedListBox());
			this->AI_Tab = (gcnew System::Windows::Forms::TabPage());
			this->xrDO = (gcnew System::Windows::Forms::TabPage());
			this->TODO = (gcnew System::Windows::Forms::TabPage());
			this->LevelsList = (gcnew System::Windows::Forms::ListBox());
			this->TabControl->SuspendLayout();
			this->Status_Tab->SuspendLayout();
			this->Geometry_Tab->SuspendLayout();
			this->SuspendLayout();
			// 
			// TabControl
			// 
			this->TabControl->Controls->Add(this->Status_Tab);
			this->TabControl->Controls->Add(this->Geometry_Tab);
			this->TabControl->Controls->Add(this->AI_Tab);
			this->TabControl->Controls->Add(this->xrDO);
			this->TabControl->Controls->Add(this->TODO);
			this->TabControl->Location = System::Drawing::Point(12, 12);
			this->TabControl->Name = L"TabControl";
			this->TabControl->SelectedIndex = 0;
			this->TabControl->Size = System::Drawing::Size(960, 674);
			this->TabControl->TabIndex = 3;
			// 
			// Status_Tab
			// 
			this->Status_Tab->BackColor = System::Drawing::Color::DimGray;
			this->Status_Tab->Controls->Add(this->button1);
			this->Status_Tab->Controls->Add(this->listBox1);
			this->Status_Tab->Font = (gcnew System::Drawing::Font(L"ScriptS_IV25", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->Status_Tab->Location = System::Drawing::Point(4, 22);
			this->Status_Tab->Name = L"Status_Tab";
			this->Status_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Status_Tab->Size = System::Drawing::Size(952, 648);
			this->Status_Tab->TabIndex = 0;
			this->Status_Tab->Text = L"Состояние компиляции";
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(667, 604);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(269, 38);
			this->button1->TabIndex = 2;
			this->button1->Text = L"Старт";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// listBox1
			// 
			this->listBox1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->listBox1->FormattingEnabled = true;
			this->listBox1->ItemHeight = 19;
			this->listBox1->Location = System::Drawing::Point(6, 17);
			this->listBox1->Name = L"listBox1";
			this->listBox1->Size = System::Drawing::Size(930, 574);
			this->listBox1->TabIndex = 0;
			// 
			// Geometry_Tab
			// 
			this->Geometry_Tab->BackColor = System::Drawing::SystemColors::WindowFrame;
			this->Geometry_Tab->Controls->Add(this->LevelsList);
			this->Geometry_Tab->Controls->Add(this->PXPM);
			this->Geometry_Tab->Controls->Add(this->label4);
			this->Geometry_Tab->Controls->Add(this->MUSamples);
			this->Geometry_Tab->Controls->Add(this->Samples);
			this->Geometry_Tab->Controls->Add(this->InfoPhases);
			this->Geometry_Tab->Controls->Add(this->progressBar1);
			this->Geometry_Tab->Controls->Add(this->label3);
			this->Geometry_Tab->Controls->Add(this->label2);
			this->Geometry_Tab->Controls->Add(this->label1);
			this->Geometry_Tab->Controls->Add(this->ThreadsCount);
			this->Geometry_Tab->Controls->Add(this->FlagsCompiler);
			this->Geometry_Tab->Location = System::Drawing::Point(4, 22);
			this->Geometry_Tab->Name = L"Geometry_Tab";
			this->Geometry_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Geometry_Tab->Size = System::Drawing::Size(952, 648);
			this->Geometry_Tab->TabIndex = 1;
			this->Geometry_Tab->Text = L"Настройка Компиляции xrLC";
			// 
			// PXPM
			// 
			this->PXPM->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->PXPM->Location = System::Drawing::Point(734, 491);
			this->PXPM->Name = L"PXPM";
			this->PXPM->Size = System::Drawing::Size(203, 29);
			this->PXPM->TabIndex = 11;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label4->Location = System::Drawing::Point(468, 493);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(250, 27);
			this->label4->TabIndex = 10;
			this->label4->Text = L"Кол-во Пикселей на метр";
			// 
			// MUSamples
			// 
			this->MUSamples->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->MUSamples->Location = System::Drawing::Point(734, 449);
			this->MUSamples->Name = L"MUSamples";
			this->MUSamples->Size = System::Drawing::Size(203, 29);
			this->MUSamples->TabIndex = 9;
			// 
			// Samples
			// 
			this->Samples->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->Samples->Location = System::Drawing::Point(734, 414);
			this->Samples->Name = L"Samples";
			this->Samples->Size = System::Drawing::Size(203, 29);
			this->Samples->TabIndex = 8;
			// 
			// InfoPhases
			// 
			this->InfoPhases->BackColor = System::Drawing::SystemColors::InfoText;
			this->InfoPhases->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->InfoPhases->ForeColor = System::Drawing::Color::Khaki;
			this->InfoPhases->FormattingEnabled = true;
			this->InfoPhases->ItemHeight = 27;
			this->InfoPhases->Location = System::Drawing::Point(6, 8);
			this->InfoPhases->Name = L"InfoPhases";
			this->InfoPhases->Size = System::Drawing::Size(251, 436);
			this->InfoPhases->TabIndex = 7;
			// 
			// progressBar1
			// 
			this->progressBar1->Location = System::Drawing::Point(9, 576);
			this->progressBar1->Name = L"progressBar1";
			this->progressBar1->Size = System::Drawing::Size(928, 54);
			this->progressBar1->TabIndex = 6;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label3->Location = System::Drawing::Point(495, 450);
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
			this->label2->Location = System::Drawing::Point(518, 414);
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
			this->label1->Location = System::Drawing::Point(518, 379);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(160, 27);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Кол-во Потоков";
			// 
			// ThreadsCount
			// 
			this->ThreadsCount->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 14.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->ThreadsCount->Location = System::Drawing::Point(734, 379);
			this->ThreadsCount->Name = L"ThreadsCount";
			this->ThreadsCount->Size = System::Drawing::Size(203, 29);
			this->ThreadsCount->TabIndex = 1;
			// 
			// FlagsCompiler
			// 
			this->FlagsCompiler->BackColor = System::Drawing::Color::White;
			this->FlagsCompiler->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 15.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->FlagsCompiler->FormattingEnabled = true;
			this->FlagsCompiler->Items->AddRange(gcnew cli::array< System::Object^  >(11) {
				L"use_embree", L"use_avx", L"use_sse", L"use_opcode_old",
					L"no_optimize", L"invalide_faces", L"nosun", L"norgb", L"nohemi", L"no_simplify", L"nosmg"
			});
			this->FlagsCompiler->Location = System::Drawing::Point(734, 17);
			this->FlagsCompiler->Name = L"FlagsCompiler";
			this->FlagsCompiler->Size = System::Drawing::Size(203, 356);
			this->FlagsCompiler->TabIndex = 0;
			// 
			// AI_Tab
			// 
			this->AI_Tab->BackColor = System::Drawing::Color::DimGray;
			this->AI_Tab->Location = System::Drawing::Point(4, 22);
			this->AI_Tab->Name = L"AI_Tab";
			this->AI_Tab->Size = System::Drawing::Size(952, 648);
			this->AI_Tab->TabIndex = 2;
			this->AI_Tab->Text = L"Настройка компиляции xrAI";
			// 
			// xrDO
			// 
			this->xrDO->BackColor = System::Drawing::Color::DimGray;
			this->xrDO->Location = System::Drawing::Point(4, 22);
			this->xrDO->Name = L"xrDO";
			this->xrDO->Size = System::Drawing::Size(952, 648);
			this->xrDO->TabIndex = 3;
			this->xrDO->Text = L"Настройка Компиляции xrDO";
			// 
			// TODO
			// 
			this->TODO->BackColor = System::Drawing::Color::DimGray;
			this->TODO->Location = System::Drawing::Point(4, 22);
			this->TODO->Name = L"TODO";
			this->TODO->Size = System::Drawing::Size(952, 648);
			this->TODO->TabIndex = 4;
			this->TODO->Text = L"TODO";
			// 
			// LevelsList
			// 
			this->LevelsList->BackColor = System::Drawing::SystemColors::HotTrack;
			this->LevelsList->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->LevelsList->ForeColor = System::Drawing::SystemColors::InactiveCaptionText;
			this->LevelsList->FormattingEnabled = true;
			this->LevelsList->ItemHeight = 27;
			this->LevelsList->Location = System::Drawing::Point(263, 8);
			this->LevelsList->Name = L"LevelsList";
			this->LevelsList->Size = System::Drawing::Size(202, 436);
			this->LevelsList->TabIndex = 12;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(984, 698);
			this->Controls->Add(this->TabControl);
			this->Name = L"MyForm";
			this->Text = L"Продвинутый компилятор Геометрии и Света (Intel)";
			this->TabControl->ResumeLayout(false);
			this->Status_Tab->ResumeLayout(false);
			this->Geometry_Tab->ResumeLayout(false);
			this->Geometry_Tab->PerformLayout();
			this->ResumeLayout(false);

		}
		
		private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e);
		
		
 
		

		public: System::Void updateLogFormItem(const char* str)
		{
			// Вызываем метод updateLog из .NET кода с использованием P/Invoke
			System::String^ managedString = gcnew System::String(str);
			listBox1->Items->Add(managedString);
		}

		public: System::Void updatePhaseItem(const char* str)
		{
			// Вызываем метод updateLog из .NET кода с использованием P/Invoke
			System::String^ managedString = gcnew System::String(str);
			InfoPhases->Items->Add(managedString);
		}

			  
 
};

}