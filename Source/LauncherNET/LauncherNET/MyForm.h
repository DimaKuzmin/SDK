#pragma once

namespace LauncherNET {

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
	private: System::Windows::Forms::ListView^ listView1;
	private: System::Windows::Forms::ListBox^ listBox1;

	private:
		/// <summary>
		/// Обязательная переменная конструктора.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Требуемый метод для поддержки конструктора — не изменяйте 
		/// содержимое этого метода с помощью редактора кода.
		/// </summary>
		void InitializeComponent(void)
		{
			this->TabControl = (gcnew System::Windows::Forms::TabControl());
			this->Status_Tab = (gcnew System::Windows::Forms::TabPage());
			this->Geometry_Tab = (gcnew System::Windows::Forms::TabPage());
			this->AI_Tab = (gcnew System::Windows::Forms::TabPage());
			this->xrDO = (gcnew System::Windows::Forms::TabPage());
			this->TODO = (gcnew System::Windows::Forms::TabPage());
			this->listBox1 = (gcnew System::Windows::Forms::ListBox());
			this->listView1 = (gcnew System::Windows::Forms::ListView());
			this->TabControl->SuspendLayout();
			this->Status_Tab->SuspendLayout();
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
			this->TabControl->Size = System::Drawing::Size(960, 568);
			this->TabControl->TabIndex = 3;
			// 
			// Status_Tab
			// 
			this->Status_Tab->BackColor = System::Drawing::Color::DimGray;
			this->Status_Tab->Controls->Add(this->listView1);
			this->Status_Tab->Controls->Add(this->listBox1);
			this->Status_Tab->Font = (gcnew System::Drawing::Font(L"ScriptS_IV25", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->Status_Tab->Location = System::Drawing::Point(4, 22);
			this->Status_Tab->Name = L"Status_Tab";
			this->Status_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Status_Tab->Size = System::Drawing::Size(952, 542);
			this->Status_Tab->TabIndex = 0;
			this->Status_Tab->Text = L"Состояние компиляции";
			// 
			// Geometry_Tab
			// 
			this->Geometry_Tab->BackColor = System::Drawing::Color::DimGray;
			this->Geometry_Tab->Location = System::Drawing::Point(4, 22);
			this->Geometry_Tab->Name = L"Geometry_Tab";
			this->Geometry_Tab->Padding = System::Windows::Forms::Padding(3);
			this->Geometry_Tab->Size = System::Drawing::Size(952, 542);
			this->Geometry_Tab->TabIndex = 1;
			this->Geometry_Tab->Text = L"Настройка Компиляции xrLC";
			// 
			// AI_Tab
			// 
			this->AI_Tab->BackColor = System::Drawing::Color::DimGray;
			this->AI_Tab->Location = System::Drawing::Point(4, 22);
			this->AI_Tab->Name = L"AI_Tab";
			this->AI_Tab->Size = System::Drawing::Size(952, 542);
			this->AI_Tab->TabIndex = 2;
			this->AI_Tab->Text = L"Настройка компиляции xrAI";
			// 
			// xrDO
			// 
			this->xrDO->BackColor = System::Drawing::Color::DimGray;
			this->xrDO->Location = System::Drawing::Point(4, 22);
			this->xrDO->Name = L"xrDO";
			this->xrDO->Size = System::Drawing::Size(952, 542);
			this->xrDO->TabIndex = 3;
			this->xrDO->Text = L"Настройка Компиляции xrDO";
			// 
			// TODO
			// 
			this->TODO->BackColor = System::Drawing::Color::DimGray;
			this->TODO->Location = System::Drawing::Point(4, 22);
			this->TODO->Name = L"TODO";
			this->TODO->Size = System::Drawing::Size(952, 542);
			this->TODO->TabIndex = 4;
			this->TODO->Text = L"TODO";
			// 
			// listBox1
			// 
			this->listBox1->Font = (gcnew System::Drawing::Font(L"Comic Sans MS", 14.25F, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Bold | System::Drawing::FontStyle::Italic)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(204)));
			this->listBox1->FormattingEnabled = true;
			this->listBox1->ItemHeight = 27;
			this->listBox1->Location = System::Drawing::Point(6, 17);
			this->listBox1->Name = L"listBox1";
			this->listBox1->Size = System::Drawing::Size(652, 355);
			this->listBox1->TabIndex = 0;
			// 
			// listView1
			// 
			this->listView1->HideSelection = false;
			this->listView1->Location = System::Drawing::Point(664, 17);
			this->listView1->Name = L"listView1";
			this->listView1->Size = System::Drawing::Size(273, 458);
			this->listView1->TabIndex = 1;
			this->listView1->UseCompatibleStateImageBehavior = false;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(984, 592);
			this->Controls->Add(this->TabControl);
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->TabControl->ResumeLayout(false);
			this->Status_Tab->ResumeLayout(false);
			this->ResumeLayout(false);

		}
 
}
