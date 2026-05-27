#include "stdafx.h"
#include "compiler.h"
#include "levelgamedef.h"
#include "level_graph.h"
#include "AIMapExport.h"
 
#include "EmbreeRayTracing.h"
//-----------------------------------------------------------------
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

void xrLoadRcast(IReader* fs);

void xrLoad(LPCSTR name, bool draft_mode)
{
	FS.get_path					("$level$")->_set	((LPSTR)name);
	string256					N;
	if (!draft_mode)	
	{
		// shaders
		string_path				N;
		FS.update_path			(N,"$game_data$","shaders_xrlc.xr");
		g_shaders_xrlc			= xr_new<Shader_xrLC_LIB> ();
		g_shaders_xrlc->Load	(N);
		 
		// Load level data
 		if (true)
		{
 			Phase("Loading Build.prj");
 
			strconcat			(sizeof(N),N,name,"build.prj");
			IReader*	fs		= FS.r_open (N);
			IReader*	F;

			// Version
			u32 version;
			fs->r_chunk			(EB_Version,&version);
			R_ASSERT			(XRCL_CURRENT_VERSION >= 17);
			R_ASSERT			(XRCL_CURRENT_VERSION <= 18);

			// Header
			b_params			Params;
			fs->r_chunk			(EB_Parameters,&Params);

			// Load level data
			transfer("materials",	g_materials,			*fs,		EB_Materials);
			transfer("shaders_xrlc",g_shader_compile,		*fs,		EB_Shaders_Compile);

			// processing geometry !
			xrLoadRcast(fs);

			// process textures
			Status			("Processing textures...");
			{
				F = fs->open_chunk	(EB_Textures);
				u32 tex_count		= F->length()/sizeof(b_texture_real);
				for (u32 t=0; t<tex_count; t++)
				{
					Progress		(float(t)/float(tex_count));

					b_texture_real		TEX;
					F->r			(&TEX,sizeof(TEX));

					b_BuildTexture	BT;
					CopyMemory		(&BT,&TEX,sizeof(TEX));

					// load thumbnail
					string128		&N = BT.name;
					LPSTR			extension = strext(N);
					if (extension)
						*extension	= 0;

					xr_strlwr		(N);
					if (0==xr_strcmp(N,"level_lods"))	{
						// HACK for merged lod textures
						BT.dwWidth	= 1024;
						BT.dwHeight	= 1024;
						BT.bHasAlpha= TRUE;
						BT.pSurface.Clear();
					} else {
						xr_strcat		(N,".thm");
						IReader* THM	= FS.r_open("$game_textures$",N);
						if (!THM)	
							continue;


						R_ASSERT2		(THM,	N);

						if (strchr(N, '.')) *(strchr(N, '.')) = 0;

						// version
						u32 version				= 0;
						R_ASSERT				(THM->r_chunk(THM_CHUNK_VERSION,&version));
						// if( version!=THM_CURRENT_VERSION )	FATAL	("Unsupported version of THM file.");

						// analyze thumbnail information
						R_ASSERT(THM->find_chunk(THM_CHUNK_TEXTUREPARAM));
						THM->r                  (&BT.THM.fmt,sizeof(STextureParams::ETFormat));
						BT.THM.flags.assign		(THM->r_u32());
						BT.THM.border_color		= THM->r_u32();
						BT.THM.fade_color		= THM->r_u32();
						BT.THM.fade_amount		= THM->r_u32();
						BT.THM.mip_filter		= THM->r_u32();
						BT.THM.width			= THM->r_u32();
						BT.THM.height           = THM->r_u32();
						BOOL			bLOD=FALSE;
						if (N[0]=='l' && N[1]=='o' && N[2]=='d' && N[3]=='\\') bLOD = TRUE;

						// load surface if it has an alpha channel or has "implicit lighting" flag
						BT.dwWidth				= BT.THM.width;
						BT.dwHeight				= BT.THM.height;
						BT.bHasAlpha			= BT.THM.HasAlphaChannel();
						BT.pSurface.Clear();
						if (!bLOD) 
						{
							if (BT.bHasAlpha || BT.THM.flags.test(STextureParams::flImplicitLighted))
							{
								clMsg("- loading: %s", N);
								string_path name;
								R_ASSERT2(Surface_Detect(name, N), "Can't load surface");
								R_ASSERT2(BT.pSurface.LoadFromFile(name), "Can't load surface");
								BT.pSurface.ClearMipLevels();
								BT.pSurface.Convert(BearTexturePixelFormat::R8G8B8A8);
								BT.pSurface.SwapRB();
								if ((BT.pSurface.GetSize().x != BT.dwWidth) || (BT.pSurface.GetSize().y != BT.dwHeight))
								{
									Msg("! THM doesn't correspond to the texture: %dx%d -> %dx%d", BT.dwWidth, BT.dwHeight, BT.pSurface.GetSize().x, BT.pSurface.GetSize().y);
									BT.dwWidth = BT.THM.width = BT.pSurface.GetSize().x;
									BT.dwHeight = BT.THM.height = BT.pSurface.GetSize().y;
								}
							} else {
								// Free surface memory
							}
						}
					}

					// save all the stuff we've created
					g_textures->push_back	(BT);
				}
			}

		}
	}
	
	// Load lights
	if (true)
	{
		strconcat				(sizeof(N),N,name,"build.prj");

		IReader*	F			= FS.r_open(N);
		R_ASSERT2				(F,"There is no file 'build.prj'!");
		IReader					&fs= *F;

		// Version
		u32 version;
		fs.r_chunk				(EB_Version,&version);
		R_ASSERT				(XRCL_CURRENT_VERSION >= 17);
		R_ASSERT				(XRCL_CURRENT_VERSION <= 18);

		// Header
		b_params				Params;
		fs.r_chunk				(EB_Parameters,&Params);

		// Lights (Static)
		{
			F = fs.open_chunk(EB_Light_static);
			b_light_static	temp;
			u32 cnt		= F->length()/sizeof(temp);
			for				(u32 i=0; i<cnt; i++)
			{
				R_Light		RL;
				F->r		(&temp,sizeof(temp));
				Flight&		L = temp.data;
				if (_abs(L.range) > 10000.f) {
					Msg		("! BAD light range : %f",L.range);
					L.range	= L.range > 0.f ? 10000.f : -10000.f;
				}

				// type
				if			(L.type == D3DLIGHT_DIRECTIONAL)	RL.type	= LT_DIRECT;
				else											RL.type = LT_POINT;

				// generic properties
				RL.position.set				(L.position);
				RL.direction.normalize_safe	(L.direction);
				RL.range				=	L.range*1.1f;
				RL.range2				=	RL.range*RL.range;
				RL.attenuation0			=	L.attenuation0;
				RL.attenuation1			=	L.attenuation1;
				RL.attenuation2			=	L.attenuation2;

				RL.amount				=	L.diffuse.magnitude_rgb	();
				RL.tri[0].set			(0,0,0);
				RL.tri[1].set			(0,0,0);
				RL.tri[2].set			(0,0,0);

				// place into layer
				if (0==temp.controller_ID)	
					g_lights.push_back		(RL);
			}
			F->close		();
		}
	}
	
	
	compiler_load_sdk_nodes(name);
}

void xrLoadRcast(IReader* fs)
{
	g_embree_faces.clear();

	Status("Loading Vertices...");
	xr_vector<Fvector> vertexs;
	{
		IReader* CHVertex = fs->open_chunk(EB_Vertices);

		u32 v_count = CHVertex->length() / sizeof(b_vertex);

		vertexs.resize(v_count);
		for (u32 i = 0; i < v_count; i++)
			CHVertex->r_fvector3(vertexs[i]);

		CHVertex->close();
	}

	//*******
	Status("Loading Faces...");
	{
		IReader* ChunkFaces = fs->open_chunk(EB_Faces);
		R_ASSERT(ChunkFaces);
		u32 f_count = ChunkFaces->length() / sizeof(b_face);

		for (u32 i = 0; i < f_count; i++)
		{
			b_face	B;
			ChunkFaces->r(&B, sizeof(B));

			FaceDataEmbree& bFace = g_embree_faces.emplace_back();
			bFace.SetFace(vertexs[B.v[0]], vertexs[B.v[1]], vertexs[B.v[2]], nullptr);
			bFace.SetMaterial(B.dwMaterial, B.dwMaterialGame, B.t);
		}
		ChunkFaces->close();
	}


	//*******
	Status("Models and References");
	IReader* MUChunk = fs->open_chunk(EB_MU_models);

	xr_map<u16, xr_vector<FaceDataEmbree>> mu_faces;
 	auto LoadMUBase = [](IReader& F, xr_vector<FaceDataEmbree>& faces)
		{
			u16 lodID;

			shared_str name;
			F.r_stringZ(name);

			// READ: vertices
			xr_vector<b_vertex>	b_vertices;
			b_vertices.resize(F.r_u32());
			F.r(&*b_vertices.begin(), (u32)b_vertices.size() * sizeof(b_vertex));

			// READ: faces
			xr_vector<b_face>	b_faces;
			b_faces.resize(F.r_u32());
			F.r(&*b_faces.begin(), (u32)b_faces.size() * sizeof(b_face));

 			// READ: lod-ID
			F.r(&lodID, 2);

			xr_vector<u32>			sm_groups;
			sm_groups.resize(b_faces.size());
			F.r(&*sm_groups.begin(), (u32)sm_groups.size() * sizeof(u32));

 			for (auto& F : b_faces)
			{
				FaceDataEmbree faceNew;
				faceNew.SetFace(b_vertices[F.v[0]], b_vertices[F.v[1]], b_vertices[F.v[2]], nullptr);
				faceNew.SetMaterial(F.dwMaterial, F.dwMaterialGame, F.t);
				faces.push_back(faceNew);
			}

			clMsg("* Loading model: '%s' - v(%d), f(%d)", *name, b_vertices.size(), b_faces.size());
		};

	if (MUChunk)
	{
		int ModelID = 0;
		while (!MUChunk->eof())
		{
			LoadMUBase(*MUChunk, mu_faces[ModelID]);
			ModelID++;
		}
		MUChunk->close();
	}

	IReader* MUChunkRef = fs->open_chunk(EB_MU_refs);
	if (MUChunkRef)
	{
		while (!MUChunkRef->eof())
		{
			b_mu_reference		R;
			MUChunkRef->r(&R, sizeof(R));

			Fmatrix xform = R.transform;				// Transformation !
			auto& faces = mu_faces[R.model_index];		// Model Buffer by Index !
			for (auto& F : faces)
			{
 				auto& F = g_embree_faces.emplace_back();

				Fvector					P[3];
				xform.transform_tiny(P[0], F.v1);
				xform.transform_tiny(P[1], F.v2);
				xform.transform_tiny(P[2], F.v3);

				F.SetFace(P[0], P[1], P[2], nullptr);
				F.SetMaterial(F.dwMaterial, F.dwMaterialGame, F.getTC0());
			}
		}
		MUChunkRef->close();
	}
 
	extern SceneEmbreeAI			 SceneEmbreeInterface;
	SceneEmbreeInterface.InitializeEmbree();
}
