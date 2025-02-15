#include "stdafx.h"
#include "UI/UIObjectList.h"
#include "SceneObject.h"
 
#include "../../XrECore/Editor/Library.h"
#include "../XrECore/Editor/EThumbnail.h"

void UIObjectList::POS_ObjectsToLTX()
{
	xr_string file;
	if (EFS.GetSaveName(_import_, file))
	{
		CInifile* ini_file = xr_new<CInifile>(file.c_str(), false, false, false);

		ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));

		ObjectList& list = ot->GetObjects();

		for (auto item : list)
		{
			if (!item->Selected())
				continue;

			Fbox box;
			item->GetBox(box);

			Fvector center;
			box.getcenter(center);

			ini_file->w_fvector3(item->GetName(), "position", item->GetPosition());
 			ini_file->w_fvector3(item->GetName(), "box_min", box.min);
			ini_file->w_fvector3(item->GetName(), "box_max", box.max);
 			ini_file->w_fvector3(item->GetName(), "box_center", center);

			CSceneObject* object = smart_cast<CSceneObject*>(item);
			if (object)
				ini_file->w_fvector3(item->GetName(), "ref_pos", object->m_pReference->ObjectXFORM().c);
		}

		ini_file->save_as(file.c_str());
	}
}

void UIObjectList::CopyTempLODforObjects()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	for (auto obj : list)
	{
		CSceneObject* object_scene = smart_cast<CSceneObject*>(obj);
		if (object_scene && object_scene->IsMUStatic())
		{
			CEditableObject* E = object_scene->GetReference();
			xr_string lod_name = E->GetLODTextureName();
			  
			string_path fn, fn_nm;
			int age, age_nm;

			FS.update_path(fn, _game_textures_, EFS.ChangeFileExt(lod_name, ".dds").c_str());
			lod_name += "_nm";
 			FS.update_path(fn_nm, _game_textures_, EFS.ChangeFileExt(lod_name, ".dds").c_str());

			if (!FS.exist(fn) || !FS.exist(fn_nm))
			{
				string_path file;
				FS.update_path(file, _import_, "TEMP_LODS\\lod_01.dds");
				if (FS.exist(file))
					FS.file_copy(file, fn);
				else
					Msg("Strange Cant Find : %s", file);
				
				Msg("Copy LOD: %s to fn: %s", file, fn);
 				
				FS.update_path(file, _import_, "TEMP_LODS\\lod_01_nm.dds");
				if (FS.exist(file))
					FS.file_copy(file, fn_nm);
				else
					Msg("Strange Cant Find : %s", file);

				Msg("Copy LOD: %s to fn: %s", file, fn_nm);
			}
			else
			{
			//	Msg("File is Exist: %s", fn);
			}

		}
	}
} 


void UIObjectList::SaveSelectedObjects()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	xr_string temp_fn = "";
	if (EFS.GetSaveName(_import_, temp_fn))
	{
		IWriter* write = FS.w_open_ex(temp_fn.c_str());

		for (auto obj : list)
		{
			CSceneObject* object_scene = smart_cast<CSceneObject*>(obj);

			if (object_scene && obj->Selected())
			{
				write->open_chunk(EOBJ_CHUNK_OBJECT_BODY);
				CEditableObject* edit_obj = object_scene->GetReference();
				edit_obj->use_global_pos = use_global_position;
 
				edit_obj->a_vPosition = obj->GetPosition();
				edit_obj->a_vRotate = obj->GetRotation();

				edit_obj->Save(*write);
				write->close_chunk();
			}

		}


		FS.w_close(write);
	}
}

void SaveFileDDS(xr_string& path, xr_string& to, char* prefix)
{
	xr_string pstr = path;
	pstr += prefix;
	pstr += ".dds";

	xr_string pexp = to;
	pexp += prefix;
	pexp += ".dds";

	if (FS.exist(pstr.c_str()))
	{
		FS.file_copy(pstr.c_str(), pexp.c_str());
		// Msg("From: %s, to Save: %s", pstr.c_str(), pexp.c_str());
	}
	else
	{
		Msg("Can't Extract: %s", pstr.c_str());
	}
}

void SaveFileTHM(xr_string& path, xr_string& to, char* prefix)
{
	xr_string pstr = path;
	pstr += prefix;
	pstr += ".thm";

	xr_string pexp = to;
	pexp += prefix;
	pexp += ".thm";


	if (FS.exist(pstr.c_str()))
	{
		FS.file_copy(pstr.c_str(), pexp.c_str());
		// Msg("Save: %s", pexp.c_str());
	}
	else
	{
		Msg("Can't Extract: %s", pstr.c_str());
	}

}


void ConstuctPath(xr_string& surface, xr_string& path_in, xr_string& path_out)
{
	string_path path, exportPath;
	FS.update_path(path, _game_textures_, "");
	FS.update_path(exportPath, _export_, "");
 
	path_in = path;
	path_in += surface.c_str();

	path_out = exportPath;
	path_out += "textures\\";
	path_out += surface.c_str();

}

void UIObjectList::ExportUsedTextures()
{
	Msg("ExportUsedTextures");

	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	xr_vector<xr_string> surface_textures;

	for (auto obj : list)
	{
		CSceneObject* sobject = smart_cast<CSceneObject*>(obj);
		if (sobject)
		{
 			for (auto surface : sobject->m_Surfaces)
			{
 				auto it = std::find_if(surface_textures.begin(), surface_textures.end(), [&](xr_string& s) {
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


	auto ParseBumpFromTexture = [&](xr_string& InFileThm, xr_string& game_textures, xr_string& out_folder)
		{

			ETextureThumbnail* pThmTexture = (ETextureThumbnail*) ImageLib.CreateThumbnail(InFileThm.c_str(), ECustomThumbnail::ETTexture);
			bool isLoaded = pThmTexture->Load(InFileThm.c_str(), 0);
			if (!isLoaded)
			{
				Msg("[Exports] Problem Load File: %s", InFileThm.c_str());
				return;
			}

			if (pThmTexture != nullptr)
			{
				shared_str Temp = *pThmTexture->_Format().bump_name;
				shared_str Detail_Map = *pThmTexture->_Format().detail_name;
 

				if (Temp.size() > 0)
				{
					{
						xr_string BumpTextureIn = game_textures + *Temp + ".dds";
						xr_string BumpTextureOut = out_folder + "\\" + *Temp + ".dds";
						FS.file_copy(BumpTextureIn.c_str(), BumpTextureOut.c_str());

						xr_string BumpTextureIn2 = game_textures + *Temp + "#.dds";
						xr_string BumpTextureOut2 = out_folder + "\\" + *Temp + "#.dds";
						FS.file_copy(BumpTextureIn2.c_str(), BumpTextureOut2.c_str());
					}

					{
						xr_string BumpTextureIn = game_textures + *Temp + ".thm";
						xr_string BumpTextureOut = out_folder + "\\" + *Temp + ".thm";
						FS.file_copy(BumpTextureIn.c_str(), BumpTextureOut.c_str());

						xr_string BumpTextureIn2 = game_textures + *Temp + "#.thm";
						xr_string BumpTextureOut2 = out_folder + "\\" + *Temp + "#.thm";
						FS.file_copy(BumpTextureIn2.c_str(), BumpTextureOut2.c_str());
					}
				}

				if (Detail_Map.size() > 0)
				{
					{
						xr_string BumpTextureIn = game_textures + *Detail_Map + ".dds";
						xr_string BumpTextureOut = out_folder + "\\" + *Detail_Map + ".dds";
						FS.file_copy(BumpTextureIn.c_str(), BumpTextureOut.c_str());

						xr_string BumpTextureIn2 = game_textures + *Detail_Map + "#.dds";
						xr_string BumpTextureOut2 = out_folder + "\\" + *Detail_Map + "#.dds";
						FS.file_copy(BumpTextureIn2.c_str(), BumpTextureOut2.c_str());
					}

					{
						xr_string BumpTextureIn = game_textures + *Detail_Map + ".thm";
						xr_string BumpTextureOut = out_folder + "\\" + *Detail_Map + ".thm";
						FS.file_copy(BumpTextureIn.c_str(), BumpTextureOut.c_str());

						xr_string BumpTextureIn2 = game_textures + *Detail_Map + "#.thm";
						xr_string BumpTextureOut2 = out_folder + "\\" + *Detail_Map + "#.thm";
						FS.file_copy(BumpTextureIn2.c_str(), BumpTextureOut2.c_str());
					}
				}
			}
		};

	auto ConstuctGamePathes = [&](xr_string& path_in, xr_string& path_out)
		{
			string_path path, exportPath;
			FS.update_path(path, _game_textures_, "");
			FS.update_path(exportPath, _export_, "");

			path_in = path;
			path_out = exportPath;
			path_out += "textures\\";
		};


	int ID = 0;
	for (auto surface : surface_textures)
	{
		xr_string game_textures, game_export;
		ConstuctGamePathes(game_textures, game_export);

		xr_string path_in, path_to;
 		ConstuctPath(surface, path_in, path_to);

		// DDS
		SaveFileDDS(path_in, path_to, "");
		// THM
		SaveFileTHM(path_in, path_to, "");

		ParseBumpFromTexture(surface, game_textures, game_export);
	

 		// // BUMP
		// SaveFileDDS(path_in, path_to, "_bump");
 		// // BUMP#
		// SaveFileDDS(path_in, path_to, "_bump#");
		
		// // BUMP
		// SaveFileTHM(path_in, path_to, "_bump");
		// // BUMP#
		// SaveFileTHM(path_in, path_to, "_bump#");
		
		ID++;
	}
}


void UIObjectList::ExportUsedObjects()
{
	ESceneCustomOTool* ot = dynamic_cast<ESceneCustomOTool*>(Scene->GetTool(LTools->CurrentClassID()));
	ObjectList& list = ot->GetObjects();

	xr_map<shared_str, int> reference;
	for (auto obj : list)
	{
		reference[obj->RefName()]++;
	}

	for (auto obj : reference)
	{
		string_path path = { 0 };;
		FS.update_path(path, _objects_, obj.first.c_str());
		xr_strcat(path, ".object");

		string_path path_E = {0};
		FS.update_path(path_E, _export_, "");
	
		xr_string tmp;
		tmp += path_E;
		tmp += "objects\\";
		tmp += obj.first.c_str();
		tmp += ".object";		 
		
		if (FS.exist(path))
		{
			FS.file_copy(path, tmp.c_str());
			Msg("Ref Copy: %s to %s", path, path_E);
		}
	}
}
