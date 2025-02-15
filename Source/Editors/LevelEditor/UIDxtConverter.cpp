#include "stdafx.h"
#include "UIDxtConverter.h"
   



UIDxtConverter* UIDxtConverter::Form_DXT = nullptr;
 

void UIDxtConverter::Show()
{

	if (Form_DXT == nullptr)
	{	
		//Msg("Show: FORM DXT");
		Form_DXT = xr_new< UIDxtConverter>();
	}
}

void UIDxtConverter::Close()
{
	//Msg("Close: FORM DXT");
	xr_delete(Form_DXT);
}

void UIDxtConverter::Update()
{ 
	//Msg("Update: FORM DXT");
	if (Form_DXT)
	{
		if (!Form_DXT->IsClosed())
		{
			Form_DXT->Draw();
		}
		else
		{
 			xr_delete(Form_DXT);
		}
	}
}



extern "C" __declspec(dllimport)
int DXTCompress	(LPCSTR out_name, u8* raw_data, u8* ext_data, u32 w, u32 h, u32 pitch, STextureParams* options, u32 depth);

#include <windows.h>
#include <shlobj.h>

char* SelectFolder()
{
	BROWSEINFO browseInfo = { 0 };
    browseInfo.hwndOwner = NULL; // Окно-владелец
    browseInfo.pidlRoot = NULL; //FS.get_path("fs_root")->m_Path; // Корневой каталог
    browseInfo.pszDisplayName = NULL; // Имя папки, выбранной пользователем
    browseInfo.lpszTitle = "Выберите папку"; // Заголовок окна выбора папки
    browseInfo.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE; // Только файловые каталоги и новый стиль диалогового окна

    LPITEMIDLIST pidl = SHBrowseForFolder(&browseInfo);
	
	char folderPath[MAX_PATH];
	if (pidl != NULL) 
	{
		
		SHGetPathFromIDList(pidl, folderPath); 

		Msg("Folder Selected: %s", folderPath);
	}

	free(pidl);

	return folderPath;
}

char* ETFormatNAMES[] =
{	
    "tfDXT1",
    "tfADXT1",
    "tfDXT3",
    "tfDXT5",
    "tf4444",
    "tf1555",
    "tf565",
    "tfRGB",
    "tfRGBA",
	"tfNVHS",
	"tfNVHU",
	"tfA8",
	"tfL8",
	"tfA8L8",
	"tfBC4",
	"tfBC5",
	"tfBC6",
	"tfBC7",
};

char* GetFormat(u32 fmt)
{
	if (fmt == -1)
		return "tfForceU32";
	else 
		return ETFormatNAMES[fmt];
}
#include "SceneObject.h"

xr_vector<xr_string> GetLevelTextures()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SCENEOBJECT));
	
	xr_vector<xr_string> surface_textures;

	if (ot)
	{
		ObjectList& list = ot->GetObjects();
		for (auto obj : list)
		{
			CSceneObject* sobject = smart_cast<CSceneObject*>(obj);
			if (sobject)
			{
				for (auto surface : sobject->m_Surfaces)
				{
					auto it = std::find_if(surface_textures.begin(), surface_textures.end(), [&](xr_string& s)
					{
						return s._Equal(surface->m_Texture.c_str());
					});
					if (it == surface_textures.end())
					{
						xr_string text = surface->m_Texture.c_str();
						surface_textures.push_back(text);
					}
				}
			}
		}

	}

	return surface_textures;
};

xr_vector<xr_string> GetSelectedOSurfacesTextures()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(OBJCLASS_SCENEOBJECT));

	xr_vector<xr_string> surface_textures;

	if (ot)
	{
		ObjectList& list = ot->GetObjects();

		for (auto obj : list)
		{
			CSceneObject* sobject = smart_cast<CSceneObject*>(obj);
			if (sobject && sobject->Selected())
			{
				for (auto surface : sobject->m_Surfaces)
				{
					auto it = std::find_if(surface_textures.begin(), surface_textures.end(), [&](xr_string& s)
						{
							return s._Equal(surface->m_Texture.c_str());
						});
					if (it == surface_textures.end())
					{
						xr_string text = surface->m_Texture.c_str();
						surface_textures.push_back(text);
					}
				}
			}
		}

	}

	return surface_textures;
};

const uint32_t DDS_MAGIC = 0x20534444;
bool SelectedOnly = false;

void UIDxtConverter::Draw()
{
	if (!ImGui::Begin("DxtConverter", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}

	ImGui::Checkbox("SelectedOnly", &SelectedOnly);

	if (ImGui::Button("THM Export (Level For Implicit)"))
	{
		string_path p;
		FS.update_path(p, "$export_folder$", "export_thms_implicit.json");

		IWriter* writer = FS.w_open(p);

		jsonxx::Object file_json;
		jsonxx::Object array_json; 

		int ID = 0;
	 
		for (auto thm : !SelectedOnly ? GetLevelTextures() : GetSelectedOSurfacesTextures())
		{
			ID++;

			Msg("Reading Texture : %s", thm.c_str());
 
 			string128 tmp_name;
			sprintf(tmp_name, "%s.thm", thm.c_str());

			string128 tmp_dds;
			sprintf(tmp_dds, "%s.dds", thm.c_str());

			string_path file_dds;
			FS.update_path(file_dds, "$game_textures$", tmp_dds);

			enum FMT_DDS {
				dxt1 = 1,
				dxt3 = 3,
				dxt5 = 5, 
				argb = 7,

				bc4_UNORM = 8,
				bc4_SNORM = 9,
				bc5_UNORM = 11, 
				bc5_SNORM = 13, 
			
				dx10 = 15,
				undefined = 0
			};

			u32 FormatDDS = 0;

			IReader* FDDS = FS.r_open(file_dds);
			if (FDDS)
			{
				uint32_t magic;
				DDS_HEADER header;

				FDDS->r(&magic, sizeof(magic));
				
				if (magic == DDS_MAGIC)
				{
					FDDS->r(reinterpret_cast<char*>(&header), sizeof(header));					 
				}

				if (header.ddspf.dwFourCC == 0x31545844) {
					FormatDDS = FMT_DDS::dxt1;
				}
				else if (header.ddspf.dwFourCC == 0x33545844) {
					FormatDDS = FMT_DDS::dxt3;
				}
				else if (header.ddspf.dwFourCC == 0x35545844)
				{
					FormatDDS = FMT_DDS::dxt5;
				}
 				else if (header.ddspf.dwFourCC == 0x55344342) { // BC4U
					FormatDDS = FMT_DDS::bc4_UNORM;
				}
				else if (header.ddspf.dwFourCC == 0x53344342) { // BC4S
					FormatDDS = FMT_DDS::bc4_SNORM;
				}
				else if (header.ddspf.dwFourCC == 0x55354342) { // BC5U
					FormatDDS = FMT_DDS::bc5_UNORM;
				}
				else if (header.ddspf.dwFourCC == 0x53354342) { // BC5S
					FormatDDS = FMT_DDS::bc5_SNORM;
				}
				else if (header.ddspf.dwFourCC == 0x30315844)
				{  
					FormatDDS = FMT_DDS::dx10;
				}
				else
					FormatDDS = FMT_DDS::undefined;

				//Msg("Readed DDS Header: Width: %u, Height: %u", header.dwWidth, header.dwHeight);

			}
			FS.r_close(FDDS);


			string_path file_dir;
			FS.update_path(file_dir, "$game_textures$", tmp_name);

			IReader* F = FS.r_open(file_dir);

			if (F)
			{
				THM thm_params;
				thm_params.ReadFromReader(F, thm.c_str());

				if (thm_params.HasImplicit() || thm_params.HasAlpha())
				{
					string128 tmp = "HasImplicit";

					if (thm_params.isDXT1())
					{
						strcat(tmp, "_DXT1_ALPHA_DEAD");
					}
					else if (thm_params.isADXT1())
					{
						strcat(tmp, "_ADXT1LP_AHA_HAS");
					}
					else if (thm_params.isDXT3())
					{
						strcat(tmp, "_DXT3");
					}
					else if (thm_params.isDXT5())
					{
						strcat(tmp, "_DXT5");
					}
					else
					{
						strcat(tmp, thm_params.CvrtFormat());
					}

					jsonxx::Object* object = 0;
					if (!array_json.has<jsonxx::Object>(tmp))
						array_json << tmp << jsonxx::Object();
					object = &array_json.get<jsonxx::Object>(tmp);

					if (object)
					{
						
						*object << thm.c_str() << thm_params.save_json(true, FormatDDS);
						
					}
				}		
				else
				{
					string128 tmp = "NotUsedImplicit";

					jsonxx::Object* object = 0;
					if (!array_json.has<jsonxx::Object>(tmp))
						array_json << tmp << jsonxx::Object();
					object = &array_json.get<jsonxx::Object>(tmp);

					if (object)
						*object << thm.c_str() << thm_params.save_json(false, FormatDDS);
				}
			}

			FS.r_close(F);
		}

		writer->w_string(array_json.json().c_str());
		FS.w_close(writer);
	}
 
	if (ImGui::Button("THM Export"))
	{
 		FS_FileSet fileset;
		string_path filepath;
		FS.update_path(filepath, "$game_textures$", "");
		FS.file_list(fileset, filepath, FS_ListFiles | FS_ClampExt, "*.thm");

  		string_path p;
		FS.update_path(p, _import_, "export_thms.json");
 
		IWriter* writer = FS.w_open(p);

		jsonxx::Object file_json;
		jsonxx::Object array_json;


		int ID = 0;
  
		for (auto thm : fileset)
		{
			ID++;
			 
			// Msg("[%d] Save To File: %s", ID, thm.name.c_str());
			string128 tmp_name;
			sprintf(tmp_name, "%s.thm", thm.name.c_str());

			string_path file_dir;
			FS.update_path(file_dir, "$game_textures$", tmp_name);

			IReader* F = FS.r_open(file_dir);

			if (F)
			{
				THM thm_params;
				thm_params.ReadFromReader(F, thm.name.c_str());
				 
				string128 tmp;
				_GetItem(thm.name.c_str(), 0, tmp, '\\');
 
				jsonxx::Object* object = 0;
				if (!array_json.has<jsonxx::Object>(tmp))
					array_json << tmp << jsonxx::Object();
 				object = &array_json.get<jsonxx::Object>(tmp);
 
				std::string group = thm_params.CvrtFormat();
				if (object)
					*object << thm.name.c_str() << thm_params.save_json(true, 0);			 
			}
 
			FS.r_close(F);	
		}
 
		writer->w_string(array_json.json().c_str());
		FS.w_close(writer);
	}

	if (ImGui::Button("Remove THM has DDS from Import"))
	{
		FS_FileSet fileset;
		string_path filepath;
		FS.update_path(filepath, _import_, "");
		FS.file_list(fileset, filepath, FS_ListFiles | FS_ClampExt, "*.thm");

		for (auto file : fileset)	
		{
			string_path path_file, path_file_dds; 
			FS.update_path(path_file, _import_, file.name.c_str());
			FS.update_path(path_file_dds, _import_, file.name.c_str());
			xr_strcat(path_file, ".thm");
 			xr_strcat(path_file_dds, ".dds");

			if (FS.exist(path_file) && FS.exist(path_file_dds) )
			{

			
				FS.file_delete(path_file);
				FS.file_delete(path_file_dds);
			}

		}
	}
  
	if (ImGui::Button("Check Formats"))
	{
		FS_FileSet fileset;
		FS.file_list(fileset, "$game_textures$", FS_ListFiles | FS_ClampExt, "*.thm");
    						  
		int i = 0;
		for (auto file : fileset)	
		{
			string_path path_file; 
			FS.update_path(path_file, "$game_textures$", file.name.c_str());
						 
			xr_strcat(path_file, ".thm");
 			
			 

 			if (FS.exist(path_file))
			{
				try
				{
					Msg("THM: %s", path_file);
					STextureParams thm;

					IReader* file_thm =  FS.r_open(path_file);
					if (file_thm)
					{
						thm.Load(*file_thm);
					}
					
					Msg("THM: %d elapsed", file_thm->elapsed());

					FS.r_close(file_thm);

					Msg("THM[%d] DATA{Format=%s, file=%s}", i, GetFormat(thm.fmt), file.name.c_str());

				}
				catch(...)
				{
					Msg("Thm %s Corupted!!!", file.name.c_str());
				}


			}
			i++;
		}
	}
	 
	if (ImGui::Button("Convert Thm to DXT5 Formats"))
	{
		FS_FileSet fileset;
		FS.file_list(fileset, "$game_textures$", FS_ListFiles | FS_ClampExt, "*.thm");

		int i = 0;
		for (auto file : fileset)
		{
			string_path path_file;
			FS.update_path(path_file, "$game_textures$", file.name.c_str());
 			xr_strcat(path_file, ".thm");

			string_path path_file_o;
			FS.update_path(path_file_o, "$export_textures$", file.name.c_str());
			xr_strcat(path_file_o, ".thm");

			STextureParams thm;

			if (FS.exist(path_file))
			{
				try
				{
					u16 version = 0;
					u32 m_Type  = 0;

					// LOAD
					{
						IReader* F = FS.r_open(path_file);
						
						if ( F->find_chunk(THM_CHUNK_VERSION) )
							version = F->r_u16();
 						

						if ( F->find_chunk(THM_CHUNK_TYPE) ) 
							m_Type = F->r_u32();
 
 						
						if (F)
							thm.Load(*F);
						FS.r_close(F);
					}
 			
					// OUT
					{
						if (m_Type != ETextureThumbnail::ETTexture)
						{
							Msg("--- Thm is Not Texture THM!!");
							continue;
						}

						IWriter* w = FS.w_open(path_file_o);
						w->open_chunk(THM_CHUNK_VERSION);
						w->w_u16(version);
						w->close_chunk();

						w->open_chunk(THM_CHUNK_TYPE);
						w->w_u32(m_Type);
						w->close_chunk();

						thm.fmt = STextureParams::tfDXT5;
						
						thm.Save(*w);
						FS.w_close(w);

						Msg("Save File : %s", path_file_o);
					} 
				}
				catch (...)
				{
					Msg("Thm %s Corupted!!!", file.name.c_str());
				}


			}
			i++;
		}
	}

	if (ImGui::Button("Convert DXT5 all"))
	{
		FS_FileSet fileset;
		FS.file_list(fileset, "$game_textures$", FS_ListFiles | FS_ClampExt, "*.dds");

		auto pb = UI->ProgressStart(fileset.size(), "Progress Tree");

		int ID = 0;
		for (auto file : fileset)
		{
			ID++;

			STextureParams fmt;

			bool exist = false;
			u16 version = 0;
			u32 m_Type = 0;


			// LOAD THM 
			{
				string_path path_filethm;
				FS.update_path(path_filethm, "$game_textures$", file.name.c_str());
				xr_strcat(path_filethm, ".thm");

				if (exist = FS.exist(path_filethm))
				{
					// LOAD
					IReader* F = FS.r_open(path_filethm);

					if (F->find_chunk(THM_CHUNK_VERSION))
						version = F->r_u16();


					if (F->find_chunk(THM_CHUNK_TYPE))
						m_Type = F->r_u32();


					if (F)
						fmt.Load(*F);
					FS.r_close(F);
				}
			}
			
			// LOAD, SAVE : DDS
			{
				string128 tmp;
				sprintf(tmp, "Texture: %s, thm Exist: %s, FMT: %s", file.name.c_str(), exist ? "true" : "false", GetFormat(fmt.fmt)); //GetFormat(fmt.fmt),
				pb->Info(tmp);
				pb->progress = ID;

				Msg(tmp);

				fmt.fmt = STextureParams::tfDXT5;

				if (!exist)
				{
					fmt.flags.set(STextureParams::flDitherColor, FALSE);
					fmt.flags.set(STextureParams::flGenerateMipMaps, FALSE);
					fmt.flags.set(STextureParams::flBinaryAlpha, FALSE);
				}

				u32 w, h;
				U32Vec				data;
				int age;
 
				string_path path_file_outdds;
				FS.update_path(path_file_outdds, "$export_textures$", file.name.c_str());
				xr_strcat(path_file_outdds, ".dds");

				if (!FS.exist(path_file_outdds))
				{
					if (ImageLib.LoadTextureData(file.name.c_str(), data, w, h, &age))
					{
						IWriter* wrrr = FS.w_open(path_file_outdds);
						FS.w_close(wrrr);

						Msg("FileSize: %d, Path DDS: %s", data.size(), path_file_outdds);

						ImageLib.Compress(path_file_outdds, (u8*)data.data(), 0, w, h, w * 4, &fmt, 4);
					}
				}
			}
			
			/*
			if (exist)
			{
				// OUT
				{
					if (m_Type != ETextureThumbnail::ETTexture)
					{
						Msg("--- Thm is Not Texture THM!!");
						continue;
					}

					string_path path_filethm;
					FS.update_path(path_filethm, "$export_textures$", file.name.c_str());
					xr_strcat(path_filethm, ".thm");
 
					IWriter* w = FS.w_open(path_filethm);
					w->open_chunk(THM_CHUNK_VERSION);
					w->w_u16(version);
					w->close_chunk();

					w->open_chunk(THM_CHUNK_TYPE);
					w->w_u32(m_Type);
					w->close_chunk();

					fmt.Save(*w);
					FS.w_close(w);

					Msg("Save File : %s", path_filethm);
				}
			}
			*/
		}

		UI->ProgressEnd(pb);
	}

	if (ImGui::Button("Check ERROR list Textures"))
	{
		auto tools = Scene->GetOTool(OBJCLASS_SCENEOBJECT);
		
		string_path p2;
		FS.update_path(p2, _import_, "errors_textures.ltx");

		CInifile* ltx = xr_new<CInifile>(p2); 
		for (auto e : tools->GetObjects())
		{
			CSceneObject* obj = smart_cast<CSceneObject*>(e);
			

			if (obj)
			{
 				for (auto surface : obj->m_Surfaces)
				{
					string_path p;
					FS.update_path(p, "$game_textures$", surface->m_Texture.c_str());
					xr_strcat(p, ".dds");

					if (!FS.exist(p))
					{
						if (!ltx->line_exist("list", p))
							ltx->w_string("list", p, "Not Exist");
					}
				}
			}
		}
		ltx->save_as();

	}
 
	if (ImGui::Button("Check And Replace SOC textures"))
	{
		string_path p;
		FS.update_path(p, _import_, "REPLACE_TEXTURES_LTX.ltx");
		CInifile* file = xr_new<CInifile>(p);

		struct Replace
		{
			shared_str first;
			shared_str second;
			shared_str section_ID;
		};

		xr_vector<Replace> sections;

		for (auto i : file->sections())
		{
			string128 name, replace;
			sprintf(name, "texture_%d", 1);
			sprintf(replace, "texture_%d_replace", 1);

			int ID = 2;
			while (i->line_exist(name))
			{				
				Replace namedata;
				namedata.first = file->r_string(i->Name.c_str(), name);
				namedata.second = file->r_string(i->Name.c_str(), replace);
				namedata.section_ID = replace;
 
				sections.push_back(namedata);
 				ID++;

				sprintf(name, "texture_%d", ID);
				sprintf(replace, "texture_%d_replace", ID);
			}
		
		}

		auto tools = Scene->GetOTool(OBJCLASS_SCENEOBJECT);

		Msg("ListSize: Replace: %d, Objects: %d", sections.size(), tools->ObjCount());

		struct CheckSurface
		{
			CSceneObject* obj;
			xr_vector<shared_str> replaces;
		};

		xr_vector<CheckSurface> objects_replaced;

		for (auto e : tools->GetObjects())
		{
			CSceneObject* obj = smart_cast<CSceneObject*>(e);
			if (obj)
			{
				bool updated = false;
				CheckSurface data;

				for (auto surface : obj->m_Surfaces)
				{
					bool finded_for_replace = false;
					shared_str replace;
					shared_str current_tex;
					shared_str texture_ID;

					for (auto sec : sections)
					{
						if (xr_strcmp(surface->m_Texture.c_str(), sec.first.c_str()) == 0)
						{
  							finded_for_replace = true; 
							
							current_tex = sec.first;
							replace = sec.second;
							texture_ID = sec.section_ID;
 
							break;
						}
					}

					if (finded_for_replace)
					{
						updated = true;
						surface->m_Texture = replace;
 						
						Msg("Replace[%s] Surface: %s to %s on onject: %s", texture_ID.c_str(), current_tex.c_str(), replace.c_str(), obj->GetName());
						data.replaces.push_back(surface->m_Texture);
					}
				}
			
				if (updated)
				{
					data.obj = obj;
					objects_replaced.push_back(data);
					obj->OnChangeSurfaces();
				}
			}
		}

		for (auto sdata : objects_replaced)
 		for (auto surface : sdata.obj->m_Surfaces)
		for (auto name : sdata.replaces)
		if (strstr(surface->m_Texture.c_str(), name.c_str()))
		{
			Msg("ReCheck Object[%s] Surface[%s]", sdata.obj->GetName(), surface->m_Texture.c_str());
			break;
		}
 	}

	ImGui::End();

}