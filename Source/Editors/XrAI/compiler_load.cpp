#include "stdafx.h"
#include "compiler.h"
//.#include "communicate.h"
#include "levelgamedef.h"
#include "level_graph.h"
#include "AIMapExport.h"

IC	const Fvector vertex_position(const CLevelGraph::CPosition &Psrc, const Fbox &bb, const SAIParams &params)
{
	Fvector				Pdest;
	int	x,z, row_length;
	row_length			= iFloor((bb.max.z - bb.min.z)/params.fPatchSize + EPS_L + 1.5f);
	x					= Psrc.xz() / row_length;
	z					= Psrc.xz() % row_length;
	Pdest.x =			float(x)*params.fPatchSize + bb.min.x;
	Pdest.y =			(float(Psrc.y())/65535)*(bb.max.y-bb.min.y) + bb.min.y;
	Pdest.z =			float(z)*params.fPatchSize + bb.min.z;
	return				(Pdest);
}

struct CNodePositionConverter {
	IC		CNodePositionConverter(const SNodePositionOld &Psrc, hdrNODES &m_header, NodePosition &np);
};

IC CNodePositionConverter::CNodePositionConverter(const SNodePositionOld &Psrc, hdrNODES &m_header, NodePosition &np)
{
	Fvector		Pdest;
	Pdest.x		= float(Psrc.x)*m_header.size;
	Pdest.y		= (float(Psrc.y)/65535)*m_header.size_y + m_header.aabb.min.y;
	Pdest.z		= float(Psrc.z)*m_header.size;
	CNodePositionCompressor(np,Pdest,m_header);
	np.y		(Psrc.y);
}

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

extern u32*		Surface_Load	(char* name, u32& w, u32& h);
extern void		Surface_Init	();
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

#include <memory>

void filter_embree_function(const struct RTCFilterFunctionNArguments* args) 
{
 
	CDB::Embree::RayQuaryStructure* ctxt = (CDB::Embree::RayQuaryStructure*) args->context;
 
	RTCHit* hit = (RTCHit*) args->hit;
	RTCRay* ray = (RTCRay*) args->ray;

	args->valid[0] = 0;
	ctxt->count++;

	b_rc_face& F								= g_rc_faces		[hit->primID];

	if (F.dwMaterial >= g_materials.size())
		Msg					("[%d] -> [%d]",F.dwMaterial, g_materials.size());

	b_material& M	= g_materials				[F.dwMaterial];
	b_texture&	T	= (*g_textures)				[M.surfidx];
	Shader_xrLCVec&	LIB = 		g_shaders_xrlc->Library	();
		
	if (M.shader_xrlc>=LIB.size())
	{
		ctxt->energy = 0;
		args->valid[0] = -1; 
		return;		 
	}

	Shader_xrLC& SH	= LIB						[M.shader_xrlc];
	if (!SH.flags.bLIGHT_CastShadow)			
	{
		ctxt->energy = 0;
		args->valid[0] = -1; 
		return;
	}

	if (T.pSurface.Empty())	
		T.bHasAlpha = FALSE;
			
	if (!T.bHasAlpha)
	{
		// Opaque poly - cache it
		//C[0].set	(rpinf.verts[0]);
		//C[1].set	(rpinf.verts[1]);
		//C[2].set	(rpinf.verts[2]);
		args->valid[0] = -1; 
		ctxt->energy = 0;
		return;
	}

	// barycentric coords
	// note: W,U,V order
	Fvector B;
	B.set	(1.0f - hit->u - hit->v, hit->u, hit->v);

	// calc UV
	Fvector2*	cuv = F.t;
	Fvector2	uv;
	uv.x = cuv[0].x*B.x + cuv[1].x*B.y + cuv[2].x*B.z;
	uv.y = cuv[0].y*B.x + cuv[1].y*B.y + cuv[2].y*B.z;

	int U = iFloor(uv.x*float(T.dwWidth) + .5f);
	int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
	U %= T.dwWidth;		if (U<0) U+=T.dwWidth;
	V %= T.dwHeight;	if (V<0) V+=T.dwHeight;

	u32 pixel		=((u32*)*T.pSurface)[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - float(pixel_a)/255.f;
	ctxt->energy	*= opac;
}

#include "cl_intersect.h"

float rayTrace	(CDB::COLLIDER* DB, Fvector& P, Fvector& D, float R/*, RayCache& C*/)
{
	R_ASSERT	(DB);

	// 1. Check cached polygon
	/*
	float _u,_v,range;
	bool res = CDB::TestRayTri(P,D,C,_u,_v,range,false);
	if (res) {
		if (range>0 && range<R) return 0;
	}
	*/

	/*
	// 2. Polygon doesn't pick - real database query
	DB->ray_query	(&Level,P,D,R);

	// 3. Analyze polygons and cache nearest if possible
	if (0==DB->r_count()) {
		return 1;
	}
	else
	{
		return getLastRP_Scale(DB,C);
	}
	*/

	RTCRayHit rayhit;
	rayhit.ray.tfar = R; 
	rayhit.ray.tnear = 0;

	rayhit.ray.org_x = P.x;
	rayhit.ray.org_y = P.y;
	rayhit.ray.org_z = P.z;

	rayhit.ray.dir_x = D.x;
	rayhit.ray.dir_y = D.y;
	rayhit.ray.dir_z = D.z;

	CDB::Embree::RayQuaryStructure data;
	data.energy = 1;

 	SceneEmbree.RayTrace(&rayhit, &data, &filter_embree_function, false);
 
	if (data.count == 0)
		return 1;

	return data.energy;
}


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

		// Load CFORM
		{
			strconcat			(sizeof(N),N,name,"build.cform");

			if (!FS.exist(N))
			{
				string_path tmp; sprintf(tmp, "No Find File: %s", N); 
				R_ASSERT2(0, tmp);
			}

			if (FS.exist(N))
			{
				IReader*			fs = FS.r_open(N);
				R_ASSERT			(fs->find_chunk(0));

				hdrCFORM			H;
				fs->r				(&H,sizeof(hdrCFORM));
				R_ASSERT			(CFORM_CURRENT_VERSION==H.version);

				Fvector*	verts	= (Fvector*)fs->pointer();
						 
				xr_vector< CDB::TRI> tris(H.facecount);
				u8* tris_pointer = (u8*)(verts + H.vertcount);
				for (u32 i = 0; i < H.facecount; i++)
				{
					memcpy(&tris[i], tris_pointer, CDB::TRI::Size());
					tris_pointer += CDB::TRI::Size();
				}				
 
				//Level.build			( verts, H.vertcount, tris.data(), H.facecount );
				//Level.syncronize	();

				CTimer t;t.Start();
 				SceneEmbree.InitGeometry(tris.data(), H.facecount, verts, H.vertcount, &filter_embree_function);
 				Msg("Loading Embree Geom: %d", t.GetElapsed_ms());

				//Msg("* Level CFORM: %dK", Level.memory()/1024);

				g_rc_faces.resize	(H.facecount);
				R_ASSERT(fs->find_chunk(1));
				fs->r				(&*g_rc_faces.begin(),g_rc_faces.size()*sizeof(b_rc_face));			
				LevelBB.set			(H.aabb);
				FS.r_close			(fs);
			}
			else 
			{
				string_path path_vs, path_tri, path_rq;
				strconcat			(sizeof(path_vs), path_vs, name,"build.cform_vs");
				strconcat			(sizeof(path_tri), path_tri, name,"build.cform_tri");
				strconcat			(sizeof(path_rq), path_rq, name,"build.cform_rq");

				bool vs = FS.exist(path_vs);
				bool rq = FS.exist(path_rq);
				bool tri = FS.exist(path_tri);

				if (vs && tri && rq)
				{
					IReader*			fs_tri = FS.r_open(path_tri);
					IReader*			fs_rq = FS.r_open(path_rq);
					IReader*			fs_vs = FS.r_open(path_vs);
				   	
					hdrCFORM			H;					
					if (fs_tri && fs_tri->find_chunk(0))
 						fs_tri->r				(&H,sizeof(hdrCFORM));			

 					xr_vector<Fvector> verts(H.vertcount);

					if (fs_vs && fs_vs->find_chunk(0))
					{
						Msg("Read VB");
						for (auto i = 0;i < H.vertcount;i++)	
							fs_vs->r(&verts[i], sizeof(Fvector));
						Msg("End VB size: %d", verts.size());
					}
										
					xr_vector< CDB::TRI> tris(H.facecount); 
					if (fs_tri && fs_tri->find_chunk(1))
					{		
						Msg("Read IB");
 						for (u32 i = 0; i < H.facecount; i++)
							fs_tri->r(&tris[i], CDB::TRI::Size());
 						Msg("End IB Size: %d", tris.size());
					}

					if (fs_rq && fs_rq->find_chunk(0))
					{
						Msg("Read RQ");
						g_rc_faces.resize	(H.facecount);
  						fs_rq->r				(&*g_rc_faces.begin(),g_rc_faces.size()*sizeof(b_rc_face));		
						Msg("End RQ size: %d", g_rc_faces.size());
					}

					FS.r_close(fs_rq);
					FS.r_close(fs_tri);
					FS.r_close(fs_vs);

					xr_delete(fs_rq);
					xr_delete(fs_tri);
					xr_delete(fs_vs);
					

					size_t commited;
					size_t free;
					size_t reserved;
					vminfo(&free, &reserved, &commited);
					Msg("Files Unload, commeted: %llu, free: %llu, reserved: %llu", 
						commited / 1024 / 1024, free / 1024 / 1024, reserved / 1024 / 1024);

					Msg("IB: %llu, VS: %llu, RQ: %llu",
						
						tris.size() * sizeof(CDB::TRI::Size()) / 1024 / 1024, 
						verts.size() * sizeof(Fvector) / 1024 / 1024, 
						g_rc_faces.size() * sizeof(b_rc_face) / 1024 / 1024 
					);

					Msg("RayQast Init");
 				    
					Level.build			( verts.data(), H.vertcount, tris.data(), H.facecount );
					Level.syncronize	();
					Msg("* Level CFORM: %dK",Level.memory()/1024);
		
					LevelBB.set			(H.aabb);
					
					Msg("RayQ Model End");

				}
				else 
				{
					Msg("!!! xrLC CFORM Check: VS: %d, RQ: %d, TRI: %d", vs, rq, tri);
				}
			}
		}

		// Load level data
		{
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
	
//	// Load emitters
//	{
//		strconcat			(N,name,"level.game");
//		IReader				*F = FS.r_open(N);
//		IReader				*O = 0;
//		if (0!=(O = F->open_chunk	(AIPOINT_CHUNK))) {
//			for (int id=0; O->find_chunk(id); id++) {
//				Emitters.push_back(Fvector());
//				O->r_fvector3	(Emitters.back());
//			}
//			O->close();
//		}
//	}
//
	// Load lights
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
				if (0==temp.controller_ID)	g_lights.push_back		(RL);
			}
			F->close		();
		}
	}

	// Init params
//	g_params.Init		();
	
	// Load initial map from the Level Editor
	{
		string_path			file_name;
		strconcat			(sizeof(file_name),file_name,name,"build.aimap");
		IReader				*F = FS.r_open(file_name);
		R_ASSERT2			(F, file_name);

		R_ASSERT			(F->open_chunk(E_AIMAP_CHUNK_VERSION));
		R_ASSERT			(F->r_u16() == E_AIMAP_VERSION);

		R_ASSERT			(F->open_chunk(E_AIMAP_CHUNK_BOX));
		F->r				(&LevelBB,sizeof(LevelBB));

		R_ASSERT			(F->open_chunk(E_AIMAP_CHUNK_PARAMS));
		F->r				(&g_params,sizeof(g_params));

		R_ASSERT			(F->open_chunk(E_AIMAP_CHUNK_NODES));
		u32					N = F->r_u32();

		R_ASSERT2			(N < ((u32(1) << u32(MAX_NODE_BIT_COUNT)) - 2),"Too many nodes!");
		
		Msg("Load Nodes Size: %d", N);
		g_nodes.resize		(N);

		hdrNODES			H;
		H.version			= XRAI_CURRENT_VERSION;
		H.count				= N+1;
		H.size				= g_params.fPatchSize;
		H.size_y			= 1.f;
		H.aabb				= LevelBB;


		string_path path;
		FS.update_path(path, "$app_data_root$", "logs\\xrAI_NodesLoad_Errors.log");
		
		IWriter* w = FS.w_open(path);
		
		for (u32 i=0; i<N; i++)
		{
 			u16 				pl;
			SNodePositionOld 	_np;
			NodePosition 		np;
			
			for (int j=0; j<4; ++j) 
			{
#ifndef _USE_NODE_POSITION_11
				u32 id = 0;
				F->r(&id, 3);
				g_nodes[i].n[j] = (*LPDWORD(&id)) & 0x00ffffff;

//				if (id > (1 << 23))
//					Msg("Check ID: %d", id);
				 
				if (id > (1 << 23))
				{
					string64 tmp;
					sprintf(tmp, "Check: Node Link: [%d], n[%d] = value: %d", i, j, id);
					w->w_string(tmp);
				}
#else 
				u32 id = F->r_u32();
 				g_nodes[i].n[j]	= id;
#endif

			}

			pl				= F->r_u16();
			pvDecompress	(g_nodes[i].Plane.n,pl);
			F->r			(&_np,sizeof(_np));
			CNodePositionConverter(_np,H,np);
			g_nodes[i].Pos	= vertex_position(np, LevelBB, g_params);

			g_nodes[i].Plane.build(g_nodes[i].Pos, g_nodes[i].Plane.n);
 
		}

		FS.w_close(w);

		Msg("Level Nodes %d", g_nodes.size());
		Msg("Level Min BB [%f][%f][%f]", LevelBB.min.x, LevelBB.min.y, LevelBB.min.z);
		Msg("Level Max BB [%f][%f][%f]", LevelBB.max.x, LevelBB.max.y, LevelBB.max.z);

		F->close			();

		/*if (!strstr(Core.Params,"-keep_temp_files"))
			DeleteFile		(file_name);*/
	}
}
