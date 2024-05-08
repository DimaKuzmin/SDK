#include "stdafx.h"
#include "UI_TreesReplacer.h"
#include "..\..\XrECore\Editor\Library.h"


UI_TreesReplacer* UI_TreesReplacer::Form_Tree = nullptr;
 

void UI_TreesReplacer::Show()
{

	if (Form_Tree == nullptr)
	{	
 		Form_Tree = xr_new< UI_TreesReplacer>();
	}
}

void UI_TreesReplacer::Close()
{
 	xr_delete(Form_Tree);
}

void UI_TreesReplacer::Update()
{ 
 	if (Form_Tree)
	{
		if (!Form_Tree->IsClosed())
		{
			Form_Tree->Draw();
		}
		else
		{
 			xr_delete(Form_Tree);
		}
	}
}

#include "CustomObject.h"


bool sort_text(shared_str a, shared_str b)
{
	// Получаем первую часть каждой строки до символа '/'
    std::string aString(a.c_str());
    std::string bString(b.c_str());
    
    std::string aFirstPart = aString.substr(0, aString.find('/'));
    std::string bFirstPart = bString.substr(0, bString.find('/'));

	return aFirstPart < bFirstPart;
}

/*
* 	Пример ListBoxFooter, Header
* 
	ImGui::ListBoxHeader("##MySpawnListBox", ImVec2(SpawnListSizeX, SpawnListSizeY));
	{
		for (int i = 0; i < sections.size(); ++i) 
		{
			if (ImGui::Selectable(sections[i].section.c_str(), current_item == i)) 
			{
				current_item = i;
			}
		}
	}
	ImGui::ListBoxFooter();

*/




bool UI_TreesReplacer::Refresh_ObjectsRef()
{
	refs_objects_vec.clear();

	ListItemsVec items;
    FS_FileSet lst;
    if (Lib.GetObjects(lst)) 
    {
        FS_FileSetIt	it = lst.begin();
        FS_FileSetIt	_E = lst.end();
        for (; it != _E; it++) 
        {
			//Msg("File: %s", it->name.c_str());
            
			if (strstr(it->name.c_str(), "trees\\") || strstr(it->name.c_str(), "new_trees\\"))
			if (strstr(it->name.c_str(), search_prefix_o))
				refs_objects_vec.push_back(it->name.c_str());
			//xr_string fn;
            //LHelper().CreateItem(items, it->name.c_str(), 0, ListItem::flDrawThumbnail, 0);
        }
    }

	return true;
}




bool UI_TreesReplacer::ReadSceneObjects_RefUsed()
{
   auto tool = Scene->GetOTool(OBJCLASS_SCENEOBJECT);
   if (tool)
   {
	   refs_vec.clear();
		auto list = tool->GetObjects();

 		xr_map<shared_str, int> refs;

		for (auto item : list)
 		   refs[item->RefName()] ++;
 
		//for (auto ref : refs)
		///	Msg("Referense: %s, cnt: %d", ref.first, ref.second);


		for (auto ref : refs)
		{
			if (strstr(ref.first.c_str(), "trees\\") || strstr(ref.first.c_str(), "new_trees\\"))
			if (strstr(ref.first.c_str(), search_prefix))
				refs_vec.push_back(ref.first);
			
		}

		//std::sort(refs_vec.begin(), refs_vec.end());
		std::sort(refs_vec.begin(), refs_vec.end(), sort_text);

		for (auto ref : refs)
		{
			if (strstr(ref.first.c_str(), "trees\\") || strstr(ref.first.c_str(), "new_trees\\"))
				Msg("REF[%s] = %d", ref.first.c_str(), ref.second);
		}
   }

   return true;
}

#include "SceneObject.h"

void UI_TreesReplacer::ReplaceLTX(bool back)
{
	string_path p ;

	FS.update_path(p, _import_, "trees_import.ltx");

	CInifile* file = xr_new<CInifile>(p);
	if (file)
	{
		bool work = true;
		int i = 0;
		
		while (work)
		{
			string128 t;
			sprintf(t, "ref_%d", i);
 
 			work = file->line_exist("trees", t);
 			bool exist_cmp = file->line_exist("trees_replace", t);

			if (work && exist_cmp)
			{
				auto tool = Scene->GetOTool(OBJCLASS_SCENEOBJECT);
 				   
				if (tool)
				{
					LPCSTR find = 0;
					LPCSTR replace = 0;

					if (!back)
					{
						find = file->r_string("trees", t);
						replace = file->r_string("trees_replace", t);
					}
					else
					{
						find = file->r_string("trees_replace", t);
						replace = file->r_string("trees", t);
					}


					Msg("Replace: [%s]: %s to [%s]: %s", t, find, t, replace);

					auto list = tool->GetObjects();

					for (auto item : list)	  
					{
						CSceneObject* sobj = (CSceneObject*) item;

						string_path fn;
						FS.update_path(fn, _objects_, EFS.ChangeFileExt(replace,".object").c_str());
						if (!FS.exist(fn))
						{
							Msg("Item Not Finded [%s]: t: %s", t, fn);
							continue;
						}

						if (strstr(item->RefName(), find) )
						{
							sobj->SetReference(replace);
						}		 
					}
				}

			}

			i++;
		}
	}
}


void UI_TreesReplacer::Draw()
{
	if (!ImGui::Begin("Trees Replacer", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}
 
	//ImGui::ListBox("##list", &current_item, refs_vec.data(), refs_vec.size(), 18);
	
	float PosX = ImGui::GetCursorPosX();
	float PosY = ImGui::GetCursorPosY();

	ImGui::Checkbox("tresslist", &ShowWindowsList);

	if (ShowWindowsList)
	{
		if (ImGui::BeginChild("#ListObjects", ImVec2(500, 500)))
		{
			if (refs_vec.size() > 0)
			{
				ImGui::ListBoxHeader("##list", ImVec2(500, 200));
				{
					for (int i = 0; i < refs_vec.size(); ++i)
					{
						if (ImGui::Selectable(refs_vec[i].c_str(), current_item == i, 0, ImVec2(350, 14)))
						{
							current_item = i;
						}
					}
				}
				ImGui::ListBoxFooter();
			}


			if (ImGui::InputText("#serch_1", search_prefix, 520))
			{
				ReadSceneObjects_RefUsed();
			}

			if (m_RealTexture_scene)
			{
				//ImGui::SetCursorPos(ImVec2(355, PosY));
				ImGui::Image(m_RealTexture_scene, ImVec2(250, 250));
			}
		}
		ImGui::EndChild();

		ImGui::SetCursorPos(ImVec2(550, PosY));

		if (ImGui::BeginChild("#ListObjects_REFS", ImVec2(500, 500)))
		{
			if (refs_objects_vec.size() > 0)
			{
				ImGui::ListBoxHeader("##list_objects", ImVec2(500, 200));
				{

					for (int i = 0; i < refs_objects_vec.size(); ++i)
					{
						if (ImGui::Selectable(refs_objects_vec[i].c_str(), current_item_object == i, 0, ImVec2(350, 14)))
						{
							current_item_object = i;
						}
					}
				}
				ImGui::ListBoxFooter();
			}

			if (ImGui::InputText("#serch_2", search_prefix_o, 520))
			{
				Refresh_ObjectsRef();
			}

			if (m_RealTexture_replace)
			{
				//ImGui::SetCursorPos(ImVec2(1100, PosY));
				ImGui::Image(m_RealTexture_replace, ImVec2(250, 250));
			}

		}
		ImGui::EndChild();
	}

	
	if (ImGui::Button("Refresh"))
	{
		ReadSceneObjects_RefUsed();		
		Refresh_ObjectsRef();
	}						  
 
	/*
	if (last_selected_object != current_item_object)
	{
		if (refs_objects_vec.size() > 0)
		{
			last_selected_object = current_item_object;
			
			auto * m_Thm = ImageLib.CreateThumbnail(refs_objects_vec[current_item_object].c_str(), EImageThumbnail::ETObject);
			m_RealTexture_replace = nullptr;
			if (m_Thm)
			{	
				m_Thm->Update(m_RealTexture_replace);	

				Msg("Update Tex:   m_RealTexture_replace : %d", m_RealTexture_replace);
			}
		}
	}

	if (last_selected != current_item)
	{
		if (refs_vec.size() > 0)
		{
			last_selected = current_item;

			auto * m_Thm = ImageLib.CreateThumbnail(refs_vec[current_item].c_str(), EImageThumbnail::ETObject);
			 m_RealTexture_scene = nullptr;
			if (m_Thm)
			{	
				m_Thm->Update(m_RealTexture_scene);	
				Msg("Update Tex:  m_RealTexture_scene : %d", m_RealTexture_scene);
 			}
		}
	}
	*/
 
	if (ImGui::Button("ExportLTX"))
	{
		if (refs_vec.size() > 0)
		{
			string_path p;
			FS.update_path(p, _import_, "TreesREF.ltx");
			CInifile* file = xr_new<CInifile>(p, false, false, false);

			if (file)
			{
				int i = 0;
 
			 
				for (auto ref : refs_vec)
				{
					string128 tmp;
					sprintf(tmp, "ref_%d", i); 

					file->w_string("trees", tmp, ref.c_str());
					i++;
				}

				for (auto ref : refs_objects_vec)
				{
					string128 tmp;
					sprintf(tmp, "ref_%d", i); 

					file->w_string("trees_refs", tmp, ref.c_str());
					i++;
				}
				 
			}

			file->save_as(p);

			
		}

		//FunctionSave();
		//FunctionSave_OBJECTS();
	}

	if (ImGui::Button("Replace"))
		ReplaceLTX(false);
	if (ImGui::Button("Replace_to_cop"))
		ReplaceLTX(true);

	ImGui::End();
}
  
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <StbImage/stb_image_write.h>

// Функция для сохранения текстуры в файл JPEG
bool SaveTextureToFile(u32* data, int width, int height, const char* filename) 
{
    // Создание массива для хранения пикселей текстуры
    	bool returned = false;
	try 
	{
		
		std::vector<unsigned char> pixels(width * height * 4); // Предполагается 32-битный формат текстуры (RGBA)

		memcpy(pixels.data(), data, width * height * 4);

		// Копирование данных текстуры в массив пикселей
		// D3D11_MAPPED_SUBRESOURCE mappedResource;
		// deviceContext->Map(texture, 0, D3D11_MAP_READ, 0, &mappedResource);
		// memcpy(pixels.data(), mappedResource.pData, pixels.size());
		// deviceContext->Unmap(texture, 0);

		// Отражение изображения по вертикали (DirectX хранит изображение в обратном порядке)
  
		std::vector<unsigned char> flippedPixels(width * height * 4);
		for (int y = 0; y < height; ++y)
		{
			memcpy(&flippedPixels[y * width * 4], &pixels[(height - y - 1) * width * 4], width * 4);
		}

		// Сохранение массива пикселей в файл JPEG
		stbi_flip_vertically_on_write(false); // Переворачиваем изображение по вертикали перед сохранением
		//returned = stbi_write_jpg(filename, width, height, 4, flippedPixels.data(), 100);
		returned = stbi_write_tga(filename, width, height, 4, flippedPixels.data());
	}
	catch(std::exception ex)
	{
		Msg("SaveTextureToFile trycatch error", ex.what());
	}

	return returned;
}
 

void UI_TreesReplacer::FunctionSave()
{
	
	STextureParams fmt;
	fmt.fmt					= STextureParams::tfDXT5;
	fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
	fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
	fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);
	

	
	int i = 0;
	for (auto ref : refs_vec)
	{ 
 
		auto * m_Thm = ImageLib.CreateThumbnail(ref.c_str(), EImageThumbnail::ETObject);

		if (m_Thm)
		{
			string128 tmp_n;
			sprintf(tmp_n, "Scene\\%s.tga", ref.c_str());
			 
			string_path file;
			FS.update_path(file, _import_, tmp_n);
			
			IWriter* w = FS.w_open(file);
			FS.w_close(w);

			//Msg("Try Export: %s", file);
			//ImageLib.Compress(file, (u8*) m_Thm->Pixels(), 0, 128, 128, 128*4, &fmt, 4);

			if (m_Thm->PixelsSize() != 16384)
			{
				Msg("Check: Pixel: %d, %s", m_Thm->PixelsSize(), ref.c_str() );
				continue;
			}

						
			STextureParams fmt;
			fmt.fmt					= STextureParams::tfDXT5;
			fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
			fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
			fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);

 			{
				m_Thm->VFlip();
				ImageLib.Compress(file, (u8*) m_Thm->Pixels(), 0, 128, 128, 128 * 4, &fmt, 4);
			}

			//if (!SaveTextureToFile(m_Thm->Pixels(), 128, 128, file))
			//	Msg("Error Export File: %s, Pixels: %d", file,  m_Thm->PixelsSize());
			//else 
			//	Msg("File Exported: %s, Pixels: %d", file,  m_Thm->PixelsSize());
		}

		i++;
		
	}

}


void UI_TreesReplacer::FunctionSave_OBJECTS()
{
	
	STextureParams fmt;
	fmt.fmt					= STextureParams::tfDXT5;
	fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
	fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
	fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);
	

	
	int i = 0;
	for (auto ref : refs_objects_vec)
	{ 
 
		auto * m_Thm = ImageLib.CreateThumbnail(ref.c_str(), EImageThumbnail::ETObject);

		if (m_Thm)
		{
			string128 tmp_n;
			sprintf(tmp_n, "Referense\\%s.dds", ref.c_str());
			 
			string_path file;
			FS.update_path(file, _import_, tmp_n);

			IWriter* w = FS.w_open(file);
			FS.w_close(w);

			if (m_Thm->PixelsSize() != 16384)
			{
				Msg("Check: Pixel: %d, %s", m_Thm->PixelsSize(), ref.c_str() );
				continue;
			}

			//Msg("Try Export: %s", file);
			//ImageLib.Compress(file, (u8*) m_Thm->Pixels(), 0, 128, 128, 128*4, &fmt, 4);

			//if (!SaveTextureToFile(m_Thm->Pixels(), 128, 128, file))
			//	Msg("Error Export File: %s, Pixels: %d", file, m_Thm->PixelsSize());
			//else 
			//	Msg("File Exported: %s, Pixels: %d", file,  m_Thm->PixelsSize());

			
			STextureParams fmt;
			fmt.fmt					= STextureParams::tfDXT5;
			fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
			fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
			fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);

 			{
				m_Thm->VFlip();
				ImageLib.Compress(file, (u8*) m_Thm->Pixels(), 0, 128, 128, 128 * 4, &fmt, 4);
			}
		}

		i++;
		
	}

}