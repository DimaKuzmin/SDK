#include "stdafx.h"
#pragma optimize( "", off )
#include "elight_def.h"



#include "build.h"

#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"


#include "../xrLCLight/xrMU_Model.h"
#include "../xrLCLight/xrMU_Model_Reference.h"

#define STB_IMAGE_IMPLEMENTATION
#include "StbImage\stb_image.h"

extern u32	version;
template <class T>
void transfer(const char *name, xr_vector<T> &dest, IReader& F, u32 chunk)
{
	IReader*	O	= F.open_chunk(chunk);
	u32		count	= O?(O->length()/sizeof(T)):0;
	clMsg			("* %16s: %d",name,count);
	if (count)  
	{
		dest.reserve(count);
		dest.insert	(dest.begin(), (T*)O->pointer(), (T*)O->pointer() + count);
	}
	if (O)		O->close	();
}


struct R_Control
{
	string64				name;
	xr_vector<u32>			data;
};
struct R_Layer
{
	R_Control				control;
	xr_vector<R_Light>		lights;
};

inline bool Surface_Detect(string_path& F, LPSTR N)
{
	FS.update_path(F, "$game_textures$", strconcat(sizeof(F), F, N, ".dds"));
	FILE* file = fopen(F, "rb");
	if (file)
	{
		fclose(file);
		return true;
	}

	return false;
}

void CBuild::CopyTexture(LPCSTR N, b_BuildTexture& BT, IWriter* w)
{
	//Msg("Cant Load THM %s", th_name);
	{
		string_path path = { 0 }, gamedata = { 0 };
		string32 name_thm = { 0 };
		sprintf(name_thm, "textures\\%s.thm", N);

		FS.update_path(path, "$xrlc_textures$", name_thm);
		FS.update_path(gamedata, _game_data_, name_thm);

		if (FS.exist(path))
		{
			FS.file_copy(path, gamedata);
			Msg("(NEED RESTART xrLC) Load THM from _import_: %s", path);
		}
		else
		{	
			Msg("Cant Find Thm in %s", path);
			string128 tmp = {0};
			sprintf(tmp, "Texture: %s.thm", N);
			w->w_string(tmp);
						 
		}
	}

	{
		string_path path = { 0 }, gamedata = { 0 };
		string32 name_file = { 0 };
		sprintf(name_file, "textures\\%s.dds", N);
		FS.update_path(path, "$xrlc_textures$", name_file);
		FS.update_path(gamedata, _game_data_, name_file);

		if (FS.exist(path))
		{
			FS.file_copy(path, gamedata);
			Msg("(NEED RESTART xrLC) Load DDS from _import_: %s", path);
		}
		else
		{	
			Msg("Cant Find DDS in %s", path);
			string128 tmp = {0};
			sprintf(tmp, "Texture: %s.dds", N);
			w->w_string(tmp);
		}
	}

	BT.dwWidth = 1024;
	BT.dwHeight = 1024;
	BT.bHasAlpha = FALSE;
	BT.THM.SetHasSurface(FALSE);
	BT.pSurface.Clear();

}

void CBuild::Load	(const b_params& Params, const IReader& _in_FS)
{
	IReader&	fs	= const_cast<IReader&>(_in_FS);
	// HANDLE		hLargeHeap	= HeapCreate(0,64*1024*1024,0);
	// clMsg		("* <LargeHeap> handle: %X",hLargeHeap);

	u32				i			= 0;

	float			p_total		= 0;
	float			p_cost		= 1.f/3.f;
	
	IReader*		F			= 0;

	// 
	string_path				sh_name;
	FS.update_path			(sh_name,"$game_data$","shaders_xrlc.xr");
	shaders().Load			(sh_name);

	Status("Start Loading Project");
	log_vminfo();


	//*******
	Status					("Vertices...");
	{
		F = fs.open_chunk		(EB_Vertices);
		u32 v_count			=	F->length()/sizeof(b_vertex);
		lc_global_data()->g_vertices().reserve		(3*v_count/2);
		scene_bb.invalidate		();
		for (i=0; i<v_count; i++)
		{
			Vertex*	pV			= lc_global_data()->create_vertex();
			F->r_fvector3		(pV->P);
			pV->N.set			(0,0,0);
			scene_bb.modify		(pV->P);
		}
		Progress			(p_total+=p_cost);
		clMsg				("* %16s: %d","vertices",lc_global_data()->g_vertices().size());
		F->close			();
	}

	log_vminfo();


	//*******
	Status					("Faces...");
	{
		F = fs.open_chunk		(EB_Faces);
		R_ASSERT				(F);
		u32 f_count			=	F->length()/sizeof(b_face);
		lc_global_data()->g_faces().reserve			(f_count);
		for (i=0; i<f_count; i++)
		{
			try 
			{
				Face*	_F			= lc_global_data()->create_face();
				b_face	B;
				F->r				(&B,sizeof(B));
				R_ASSERT			(B.dwMaterialGame<65536);

				_F->dwMaterial		= u16(B.dwMaterial);
				_F->dwMaterialGame	= B.dwMaterialGame;

				// Vertices and adjacement info
				for (u32 it=0; it<3; ++it)
				{
					int id			= B.v[it];
					R_ASSERT		(id<(int)lc_global_data()->g_vertices().size());
					_F->SetVertex	(it,lc_global_data()->g_vertices()[id]);
				}

				// transfer TC
				Fvector2				uv1,uv2,uv3;
				uv1.set				(B.t[0].x,B.t[0].y);
				uv2.set				(B.t[1].x,B.t[1].y);
				uv3.set				(B.t[2].x,B.t[2].y);
				_F->AddChannel		( uv1, uv2, uv3 );
			} 
			catch (...)
			{
				err_save	();
				Debug.fatal	(DEBUG_INFO,"* ERROR: Can't process face #%d",i);
			}
		}
		Progress			(p_total+=p_cost);
		clMsg				("* %16s: %d","faces",lc_global_data()->g_faces().size());
		F->close			();

		if(g_using_smooth_groups)
		{
			F = fs.open_chunk		(EB_SmoothGroups);
			
			R_ASSERT2				(F,"EB_SmoothGroups chunk not found.");
			
			u32* sm_groups			= NULL;
			u32 sm_count			=	F->length()/sizeof(u32);

			R_ASSERT				( sm_count == lc_global_data()->g_faces().size() );
			sm_groups				= xr_alloc<u32>(sm_count);
			F->r					(sm_groups, F->length());
			F->close				();

			for(u32 idx=0; idx<sm_count; ++idx)
				lc_global_data()->g_faces()[idx]->sm_group = sm_groups[idx];

			xr_free					(sm_groups);
		}
		
		if (InvalideFaces())	
		{
			err_save		();
			Debug.fatal		(DEBUG_INFO,"* FATAL: %d invalid faces. Compilation aborted",InvalideFaces());
		}
	}

	log_vminfo();


	//*******
	Status	("Models and References");
	F = fs.open_chunk		(EB_MU_models);
	if (F)
	{
		while (!F->eof())
		{
			mu_models().push_back				(xr_new<xrMU_Model>());
			mu_models().back()->Load			(*F, version );
		}
		F->close				();
	}
	F = fs.open_chunk		(EB_MU_refs);
	if (F)
	{
		while (!F->eof())
		{
			mu_refs().push_back				(xr_new<xrMU_Reference>());
			mu_refs().back()->Load			( *F, mu_models() );
		}		
		F->close				();
	}

	log_vminfo();


	//*******
	Status	("Other transfer...");
	transfer("materials",	materials(),			fs,		EB_Materials);
	transfer("shaders",		shader_render,		fs,		EB_Shaders_Render);
	transfer("shaders_xrlc",shader_compile,		fs,		EB_Shaders_Compile);
	transfer("glows",		glows,				fs,		EB_Glows);
	transfer("portals",		portals,			fs,		EB_Portals);
	transfer("LODs",		lods,				fs,		EB_LOD_models);

	// Load lights
	Status	("Loading lights...");
	{
		xr_vector<R_Layer>			L_layers;
		xr_vector<BYTE>				L_control_data;

		// Controlles/Layers
		{
			F = fs.open_chunk		(EB_Light_control);
			L_control_data.assign	(LPBYTE(F->pointer()),LPBYTE(F->pointer())+F->length());

			R_Layer					temp;

			while (!F->eof())
			{
				F->r				(temp.control.name,sizeof(temp.control.name));
				u32 cnt				= F->r_u32();
				temp.control.data.resize(cnt);
				F->r				(&*temp.control.data.begin(),cnt*sizeof(u32));

				L_layers.push_back	(temp);
			}

			F->close		();
		}
		// Static
		{
			F = fs.open_chunk	(EB_Light_static);
			b_light_static		temp;
			u32 cnt				= F->length()/sizeof(temp);
			for	(i=0; i<cnt; i++)
			{
				R_Light		RL;
				F->r		(&temp, sizeof(temp));
				Flight	L	= temp.data;

				// type
				if			(L.type == D3DLIGHT_DIRECTIONAL)	RL.type	= LT_DIRECT;
				else											
					RL.type = LT_POINT;
				RL.level	= 0;

				// split energy/color
				float			_e		=	(L.diffuse.r+L.diffuse.g+L.diffuse.b)/3.f;
				Fvector			_c		=	{L.diffuse.r,L.diffuse.g,L.diffuse.b};
				if (_abs(_e)>EPS_S)		_c.div	(_e);
				else					{ _c.set(0,0,0); _e=0; }

				// generic properties
				RL.diffuse.set				(_c);
				RL.position.set				(L.position);
				RL.direction.normalize_safe	(L.direction);
				RL.range				=	L.range*1.1f;
				RL.range2				=	RL.range*RL.range;
				RL.attenuation0			=	L.attenuation0;
				RL.attenuation1			=	L.attenuation1;
				RL.attenuation2			=	L.attenuation2;
				RL.falloff				=   1.0f/(RL.range*(RL.attenuation0 + RL.attenuation1*RL.range + RL.attenuation2*RL.range2));
				RL.energy				=	_e;

				// place into layer
				R_ASSERT	(temp.controller_ID<L_layers.size());
				L_layers	[temp.controller_ID].lights.push_back	(RL);
			}
			F->close		();
		}

		// ***Search LAYERS***
		for (u32 LH=0; LH<L_layers.size(); LH++)
		{
			R_Layer&	TEST	= L_layers[LH];
			if (0==stricmp(TEST.control.name,LCONTROL_HEMI))
			{
				// Hemi found
				L_static().hemi			= TEST.lights;
			}
			if (0==stricmp(TEST.control.name,LCONTROL_SUN))
			{
				// Sun found
				L_static().sun			= TEST.lights;
			}
			if (0==stricmp(TEST.control.name,LCONTROL_STATIC))
			{
				// Static found
				L_static().rgb			= TEST.lights;
			}
		}

		clMsg	("*lighting*: HEMI:   %d lights",L_static().hemi.size());
		clMsg	("*lighting*: SUN:    %d lights",L_static().sun.size());
		clMsg	("*lighting*: STATIC: %d lights",L_static().rgb.size());

		R_ASSERT(L_static().hemi.size());
		R_ASSERT(L_static().sun.size());
		R_ASSERT(L_static().rgb.size());

		// Dynamic
		transfer("d-lights",	L_dynamic,			fs,		EB_Light_dynamic);
	}

	log_vminfo();


 	string_path path;
	string128 name = {0};
	sprintf(name, "dump_data\\%s", "xrlc_error_textures.dump");
	FS.update_path(path, "$logs$", name);
 
	IWriter* w = FS.w_open(path);
	
	// process textures
	Status			("Processing textures...");
	{
		F = fs.open_chunk	(EB_Textures);
		u32 tex_count	= F->length()/sizeof(b_texture_real);
		for (u32 t=0; t<tex_count; t++)
		{
			Progress		(float(t)/float(tex_count));

			b_texture_real		TEX;
			F->r			(&TEX,sizeof(TEX));

			b_BuildTexture	BT;
			CopyMemory		(&BT,&TEX,sizeof(TEX));

			// load thumbnail
			LPSTR N			= BT.name;
			if (strchr(N,'.')) *(strchr(N,'.')) = 0;
			strlwr			(N);
			
			if (0==xr_strcmp(N,"level_lods"))	
			{
				// HACK for merged lod textures
				BT.dwWidth		= 1024;
				BT.dwHeight		= 1024;
				BT.bHasAlpha	= TRUE;
				BT.THM.SetHasSurface(FALSE);
				BT.pSurface.Clear();

			} 
			else
			{

				string_path			th_name;
				FS.update_path	(th_name,"$game_textures$",strconcat(sizeof(th_name),th_name,N,".thm"));
				clMsg			("processing: %s",th_name);
				IReader* THM	= FS.r_open(th_name);
				 
				if (!THM)
				{
					CopyTexture(N, BT, w);
				}
				else
				{
					R_ASSERT2(THM, th_name);

					// version
					u32 version = 0;
					R_ASSERT2(THM->r_chunk(THM_CHUNK_VERSION, &version), th_name);
					// if( version!=THM_CURRENT_VERSION )	FATAL	("Unsupported version of THM file.");

					// analyze thumbnail information
					R_ASSERT2(THM->find_chunk(THM_CHUNK_TEXTUREPARAM), th_name);
					THM->r(&BT.THM.fmt, sizeof(STextureParams::ETFormat));
					BT.THM.flags.assign(THM->r_u32());
					BT.THM.border_color = THM->r_u32();
					BT.THM.fade_color = THM->r_u32();
					BT.THM.fade_amount = THM->r_u32();
					BT.THM.mip_filter = THM->r_u32();
					BT.THM.width = THM->r_u32();
					BT.THM.height = THM->r_u32();
					BOOL			bLOD = FALSE;
					if (N[0] == 'l' && N[1] == 'o' && N[2] == 'd' && N[3] == '\\') bLOD = TRUE;

					// load surface if it has an alpha channel or has "implicit lighting" flag
					BT.dwWidth = BT.THM.width;
					BT.dwHeight = BT.THM.height;
					BT.bHasAlpha = BT.THM.HasAlphaChannel();
 
					if (!bLOD)
					{
						if (BT.bHasAlpha || BT.THM.flags.test(STextureParams::flImplicitLighted) || g_build_options.b_radiosity)
						{
							clMsg("- loading: %s", N);
							string_path name;
							
							if (!Surface_Detect(name, N) || !BT.pSurface.LoadFromFile(name))
							{
								Msg("Not Find Texture, Try Copy: %s", N);
								CopyTexture(N, BT, w);

								string128 tmp = {0};
								sprintf(tmp, "Texture: %s.dds", N);
								w->w_string(tmp);
								continue;
							}

							//R_ASSERT2(Surface_Detect(name, N), "Can't load surface");
							//R_ASSERT2(BT.pSurface.LoadFromFile(name), "Can't load surface");

							BT.pSurface.ClearMipLevels();
							BT.THM.SetHasSurface(true);
							BT.pSurface.Convert(BearTexturePixelFormat::R8G8B8A8);
							BT.pSurface.SwapRB();

							if ((BT.pSurface.GetSize().x != BT.dwWidth) || (BT.pSurface.GetSize().y != BT.dwHeight))
							{
								
								string256 msg;
								sprintf(msg, "! THM doesn't correspond to the texture: %dx%d -> %dx%d, texture: %s", BT.dwWidth, BT.dwHeight, BT.pSurface.GetSize().x, BT.pSurface.GetSize().y, N);
								Msg(msg);
								w->w_string(msg);

								BT.dwWidth = BT.THM.width = BT.pSurface.GetSize().x;
								BT.dwHeight = BT.THM.height = BT.pSurface.GetSize().y;
							}

							//BT.Vflip	();
						}
					}
				}

				
			}

			BOOL			bLOD = FALSE;
			if (N[0] == 'l' && N[1] == 'o' && N[2] == 'd' && N[3] == '\\') bLOD = TRUE;

			/*
			if (BT.THM.flags.test(STextureParams::flImplicitLighted) && !bLOD)
			{	
				if (strstr(Core.Params, "-cvrt_impl_texture_2"))
				{
					BT.pSurface.Scale(BT.dwWidth / 2, BT.dwHeight / 2);
					BT.dwWidth = BT.dwWidth / 2;
					BT.dwHeight = BT.dwHeight / 2;
				}

				if (strstr(Core.Params, "-cvrt_impl_texture_4"))
				{
					BT.pSurface.Scale(BT.dwWidth / 4, BT.dwHeight / 4);
					BT.dwWidth = BT.dwWidth / 4;
					BT.dwHeight = BT.dwHeight / 4;
				}
				 
				if (strstr(Core.Params, "-cvrt_impl_texture_8"))
				{
					BT.pSurface.Scale(BT.dwWidth / 8, BT.dwHeight / 8);
					BT.dwWidth = BT.dwWidth / 8;
					BT.dwHeight = BT.dwHeight / 8;
				}

				if (strstr(Core.Params, "-cvrt_impl_texture_16"))
				{
					BT.pSurface.Scale(BT.dwWidth / 16, BT.dwHeight / 16);
					BT.dwWidth = BT.dwWidth / 16;
					BT.dwHeight = BT.dwHeight / 16;
				}


				if (strstr(Core.Params, "-cvrt_impl_texture_128"))
				{
					BT.pSurface.Scale(128, 128);
					BT.dwWidth = 128;
					BT.dwHeight = 128;
				}
			}	
			*/

			if (strstr(Core.Params, "-cvrt_impl_1024"))
			if (BT.dwWidth > 1024 || BT.dwHeight > 1024)
			{
				BT.dwWidth = 1024;
				BT.dwHeight = 1024;
			}


			// save all the stuff we've created
			textures().push_back	(BT);
		}
	}

	// post-process materials
	Status	("Post-process materials...");
	post_process_materials( shaders(), shader_compile, materials() );

	Progress(p_total+=p_cost);

	// Parameter block
	CopyMemory(&g_params(),&Params,sizeof(b_params));

	FS.w_close(w);

	log_vminfo();

	// 
	clMsg	("* sizes: V(%d),F(%d)",sizeof(Vertex),sizeof(Face));
}



