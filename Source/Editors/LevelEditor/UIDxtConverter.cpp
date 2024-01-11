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

void UIDxtConverter::Draw()
{
	if (!ImGui::Begin("DxtConverter", &bOpen))	// ImGuiWindowFlags_NoResize
	{
		ImGui::PopStyleVar(1);
		ImGui::End();
		return;
	}
	/*
	if (ImGui::Button("Select To Convert"))
	{
		xr_strcpy(path_dir, SelectFolder());

		//xr_string name, path; 
		//if (EFS.GetOpenPathName(EDevice.m_hWnd, "$fs_root$", path, name))
 		//	xr_strcpy(path_dir,path.c_str());
 	}

 	ImGui::Text("SelectDir: %s", path_dir);

	if (ImGui::Button("Select To Out"))
	{	 
		xr_strcpy(path_dir_out, SelectFolder());
	}
	*/

	/*
					    ETFormat	        fmt;
						Flags32		        flags;
						u32			        border_color;
						u32			        fade_color;
						u32			        fade_amount;
						u8					fade_delay;
						u32			        mip_filter;
						int			        width;
						int			        height;
						// detail ext
						shared_str			detail_name;
						float		        detail_scale;
						ETType		        type;
						// material
						ETMaterial			material;
						float				material_weight;
						// bump	
						float 				bump_virtual_height;
						ETBumpMode			bump_mode;
						shared_str			bump_name;
						shared_str			ext_normal_map_name;
				*/

	FS.update_path(path_dir, "$convert_textures$", "");
 	FS.update_path(path_dir_out, "$export_textures$", "");

	ImGui::Text("TexturesDir: %s", path_dir);
 	ImGui::Text("OutDir: %s", path_dir_out);

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
		FS.file_list(fileset, path_dir, FS_ListFiles | FS_ClampExt, "*.thm");
    						  
		int i = 0;
		for (auto file : fileset)	
		{
			string_path path_file; 
			FS.update_path(path_file, "$convert_textures$", file.name.c_str());
						 
			xr_strcat(path_file, ".thm");
 			
			STextureParams thm;

 			if (FS.exist(path_file))
			{
				try
				{
					Msg("THM: %s", path_file);

					IReader* file_thm =  FS.r_open(path_file);
					if (file_thm)
					{
						thm.Load(*file_thm);
					}
					
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

	if (ImGui::Button("Convert DXT"))
	{
		FS_FileSet fileset;
		FS.file_list(fileset, path_dir, FS_ListFiles | FS_ClampExt, "*.dds");
    
		for (auto file : fileset)
		{
			 
			string_path path_file; 
			FS.update_path(path_file, "$convert_textures$", file.name.c_str());
			
			string_path path_file_out; 
			FS.update_path(path_file_out, "$export_textures$", file.name.c_str());
			 
			xr_strcat(path_file, ".thm");
			xr_strcat(path_file_out, ".thm");
		

			//bool loaded = false;
			/*
			if (FS.exist(path_file))
			{
				ETextureThumbnail thm(path_file); 
				thm._Format().fmt = STextureParams::ETFormat::tfDXT5;
 				thm.Save(0, path_file_out);
 
				loaded = true;
			}
			*/
		    
			string_path path_filedds; 
			FS.update_path(path_filedds, "$convert_textures$", file.name.c_str());
			
			string_path path_file_outdds; 
			FS.update_path(path_file_outdds, "$export_textures$", file.name.c_str());

			xr_strcat(path_filedds, ".dds");
			xr_strcat(path_file_outdds, ".dds");

			if (FS.exist(path_filedds) )
			{	
				
				STextureParams fmt;
				bool exist = false;
				if (exist = FS.exist(path_file))
				{
					
				}
				else 
				{
 					fmt.fmt					= STextureParams::tfDXT5;
					fmt.flags.set			(STextureParams::flDitherColor,		FALSE);
					fmt.flags.set			(STextureParams::flGenerateMipMaps,	FALSE);
					fmt.flags.set			(STextureParams::flBinaryAlpha,		FALSE);
				}
 
				ETextureThumbnail thm(path_file); 
				thm._Format().fmt = STextureParams::ETFormat::tfDXT5;
 				thm.Save(0, path_file_out);

				Msg("Texture: %s, format: %s", path_filedds, GetFormat(fmt.fmt) );
			   	u32 w, h;
				U32Vec				data;
				int age; 
 
 
				if (ImageLib.LoadTextureData( file.name.c_str(), data, w, h, &age))
				{
					IWriter* wrr = FS.w_open(path_file_outdds);
					FS.w_close(wrr);

					if (exist)
					{
						ETextureThumbnail thm(path_file); 
						thm._Format().fmt = STextureParams::ETFormat::tfDXT5;
 						thm.Save(0, path_file_out);

						ImageLib.Compress(path_file_outdds, (u8*) data.data(), 0, w, h, w * 4, &thm._Format(), 4);

					}
					else 
					{
						ImageLib.Compress(path_file_outdds, (u8*) data.data(), 0, w, h, w * 4,  &fmt, 4);
					}


				}			 
			}


			 
			
		}
		 

		
	}


	ImGui::End();
}