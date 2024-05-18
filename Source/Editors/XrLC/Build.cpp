// Build.cpp: implementation of the CBuild class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "build.h"

#include "../xrLCLight/xrMU_Model.h"



#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"
#include "../xrLCLight/mu_model_light.h"
 
//#include "../xrLCLight/lcnet_task_manager.h"
void	calc_ogf		( xrMU_Model &	mu_model );
void	export_geometry	( xrMU_Model &	mu_model );

void	export_ogf		( xrMU_Reference& mu_reference );



#include "../XrLCLight/BuildArgs.h"

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;

using namespace			std;
struct OGF_Base;
SBuildOptions			g_build_options;

xr_vector<OGF_Base *>	g_tree;
vec2Face				g_XSplit;


//BOOL					b_noise		= FALSE;
//BOOL					b_radiosity	= FALSE;
//BOOL					b_net_light	= FALSE;


 
void	CBuild::CheckBeforeSave( u32 stage )
{
	bool b_g_tree_empty = g_tree.empty() ;
	R_ASSERT( b_g_tree_empty );
	bool b_g_XSplit_empty = g_XSplit.empty();
	R_ASSERT( b_g_XSplit_empty );
	bool b_IsOGFContainersEmpty = IsOGFContainersEmpty();
	R_ASSERT( b_IsOGFContainersEmpty );
	
	
	
}

void	CBuild::TempSave( u32 stage )
{
	CheckBeforeSave( stage );

}

//////////////////////////////////////////////////////////////////////

CBuild::CBuild()
{
	

}

CBuild::~CBuild()
{
	destroy_global_data();
}
 
CMemoryWriter&	CBuild::err_invalid()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_invalid(); 
}
CMemoryWriter&	CBuild::err_multiedge()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_multiedge(); 
}
CMemoryWriter	&CBuild::err_tjunction()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->err_tjunction(); 
}
xr_vector<b_material>&	CBuild::materials()	
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->materials(); 
}
xr_vector<b_BuildTexture>&	CBuild::textures()		
{
	VERIFY(lc_global_data());
	return lc_global_data()->textures(); 
}

base_lighting&	CBuild::L_static()
{
	VERIFY(lc_global_data()); return lc_global_data()->L_static(); 
}

Shader_xrLC_LIB&	CBuild::shaders()		
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->shaders(); 
}

extern u16		RegisterShader		(LPCSTR T);


void CBuild::Light_prepare()
{
	for (vecFaceIt I=lc_global_data()->g_faces().begin();	I!=lc_global_data()->g_faces().end(); I++) (*I)->CacheOpacity();
	for (u32 m=0; m<mu_models().size(); m++)	mu_models()[m]->calc_faceopacity();
}

#ifdef LOAD_GL_DATA
void net_light ();
#endif

 
extern string_path LEVEL_PATH;

#include "..\XrLCLight\xrHardwareLight.h"


void CBuild::TestMergeGeom(IWriter* writer)
{

	Phase("TEST MERGE");
	
	//FPU::m64r();
	//Phase("Optimizing...");
	//mem_Compact();
	//CorrectTJunctions();

	//****************************************** HEMI-Tesselate
	/*
	FPU::m64r();
	Phase("Adaptive HT...");
	mem_Compact();
	log_vminfo();
	xrPhase_AdaptiveHT();
	*/

	FPU::m64r();
	Phase("Building normals...");
	mem_Compact();
	CalcNormals();
	//SmoothVertColors			(5);

	FPU::m64r					();
	Phase						("Building collision database...");
	mem_Compact					();

	
	BuildCForm					();
  	BuildPortals(*writer);

	log_vminfo();

	//****************************************** T-Basis
	
	{
		FPU::m64r();
		Phase("Building tangent-basis...");
		xrPhase_TangentBasis();
		mem_Compact();
	}

	log_vminfo();

 
	//****************************************** GLOBAL-RayCast model
	FPU::m64r();
	Phase("Building rcast-CFORM model...");
	mem_Compact();
	Light_prepare();

	log_vminfo();


	CTimer t;t.Start();
	BuildRapid(TRUE);
	Status("RcastModel LoadTime: %d", t.GetElapsed_ms());

	log_vminfo();

 
	FPU::m64r					();
	Phase						("Resolving materials...");
	mem_Compact					();
	xrPhase_ResolveMaterials	();
	IsolateVertices				(TRUE);

	log_vminfo();


	//****************************************** UV mapping
	{
		FPU::m64r					();
		Phase						("Build UV mapping...");
		
		mem_Compact					();
 		xrPhase_UVmap				();
		IsolateVertices				(TRUE);
	}
	
	log_vminfo();

	//****************************************** Subdivide geometry
	FPU::m64r					();
	Phase						("Subdividing geometry...");
	
	mem_Compact					();
 	xrPhase_Subdivide			();

	log_vminfo();

	//IsolateVertices				(TRUE);
	Phase						("Clear Isolated Vertex Pool...");
	lc_global_data()->vertices_isolate_and_pool_reload();

	log_vminfo();
  
	/*
	
	PART 2
	-----------------------------------------------------------------------------------------------------
	
	*/

	//****************************************** Merge geometry
	FPU::m64r					();
	Phase						("Merging geometry...");
	mem_Compact					();
	xrPhase_MergeGeometry		();

	//****************************************** Convert to OGF
	FPU::m64r					();
	Phase						("Converting to OGFs...");
	mem_Compact					();
	Flex2OGF					();
 
	//****************************************** Export MU-models
	FPU::m64r					();
	Phase						("Converting MU-models to OGFs...");
	mem_Compact					();
	{
		u32 m;
		Status			("MU : Models...");
		for (m=0; m<mu_models().size(); m++)	
		{
			clMsg("ID[%d], size[%d]", m, mu_models().size());
			calc_ogf			(*mu_models()[m]);
			export_geometry		(*mu_models()[m]);
		}

		Status			("MU : References...");
		for (m = 0; m < mu_refs().size(); m++)
		{
			clMsg("muref ID[%d], size[%d]", m, mu_models().size());
			export_ogf(*mu_refs()[m]);
		}
	}

	//****************************************** Destroy RCast-model
	FPU::m64r		();
	Phase			("Destroying ray-trace model...");
	mem_Compact		();
	lc_global_data()->destroy_rcmodel();

	//****************************************** Build sectors
	FPU::m64r();
	Phase("Building sectors...");
	mem_Compact();
	BuildSectors();
 
	//****************************************** Saving MISC stuff
	FPU::m64r		();
	Phase			("Saving...");
	mem_Compact		();
	SaveLights		(*writer);

	writer->open_chunk	(fsL_GLOWS);
	
	for (u32 i=0; i<glows.size(); i++)
	{
		b_glow&	G	= glows[i];
		writer->w		(&G,4*sizeof(float));
		string1024	sid;
		strconcat	(sizeof(sid),sid,
			shader_render[materials()[G.dwMaterial].shader].name,
			"/",
			textures()		[materials()[G.dwMaterial].surfidx].name
			);
		writer->w_u16	(RegisterShader(sid));
	}
	writer->close_chunk	();

	SaveTREE		(*writer);
	SaveSectors		(*writer);

	err_save		();
 
	mem_Compact();

}

void CBuild::ExportRayCastModel(IWriter* writer)
{
	Phase("Export XFORM, RAYCAST GEOM");
 
	FPU::m64r();
	Phase("Building normals...");
	mem_Compact();
	CalcNormals();
 
	FPU::m64r					();
	Phase						("Building collision database...");
	mem_Compact					();
	BuildCForm					();
 
	BuildPortals(*writer);

	//****************************************** T-Basis
	
	{
		FPU::m64r();
		Phase("Building tangent-basis...");
		xrPhase_TangentBasis();
		mem_Compact();
	}
 
	//****************************************** GLOBAL-RayCast model
	FPU::m64r();
	Phase("Building rcast-CFORM model...");
	mem_Compact();
	Light_prepare();

	CTimer t;t.Start();
	BuildRapid(TRUE);
	Msg("RcastModel LoadTime: %d", t.GetElapsed_ms());
}

#include "../XrLCLight/xrDeflector.h"
/*
void CBuild::ExportDeflectors()
{
	Phase("Export Data For Lighting");

	string256 s, sx;  
 
	IWriter* write = FS.w_open(strconcat(sizeof(s), s, pBuild->path, "build.deflectors"));
	
	write->open_chunk(0);
 
	int id = 0;
	for (Face* faces : lc_global_data()->g_faces())
	{
		faces->set_index(id);
 		id++;
	}

	Status("SizeDeflectors: %d", lc_global_data()->g_deflectors().size());

	for (auto defl : lc_global_data()->g_deflectors())
	{
		defl->Serialize(write);	
	}
 	write->close_chunk();
 
	 
	IWriter* write_x = FS.w_open(strconcat(sizeof(sx), sx, pBuild->path, "build.splitx"));

	write_x->open_chunk(0);
			
	write_x->w_u32(g_XSplit.size());
	for (auto vec : g_XSplit)
	{
		write_x->w_u32(vec->size());
		for (auto face : *vec)
			write_x->w_u32(face->self_index());
	}
	write_x->close_chunk();
	
	Status("Size GXSplit: %d", g_XSplit.size());

	FS.w_close(write_x);
	FS.w_close(write); 
}
*/

IC bool				FaceEqual(Face& F1, Face& F2);
#include "../XrLCLight/xrMU_Model_Reference.h"
#include <iostream>
#include <fstream>			 

struct VertexMAP
{
	int v1,v2,v3;
};

void SaveToBin(size_t v_count, Fvector* verts, size_t t_count, std::vector<VertexMAP>& mapindexes, char* filename)
{
	string_path p_sdk ;
	string256 ff;
	sprintf(ff, "worldobj\\chunks_bin\\%s",filename);

	FS.update_path(p_sdk, "$fs_root$", ff);

	Status("Save File: %s", ff);
	
	 


	IWriter* write = FS.w_open(ff);
 
    // Записываем вершины
	write->w_u32(v_count);
	for (int i = 0; i < v_count;i ++)
	{	
 
		if (i % 1024 == 0)
		StatusNoMSG("Size: %d/%d", i, v_count);
        //file << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
		write->w_fvector3(verts[i]);
    }

    // Записываем индексы 
	write->w_u32(t_count);
	for (int i = 0; i < t_count; i ++) 
	{
		 
		if (i % 1024 == 0)
			StatusNoMSG("Size: %d/%d", i, t_count);
        //file << "f " << mapindexes[i].v1 + 1 << " " << mapindexes[i].v2 + 1 << " " << mapindexes[i].v3 + 1 << "\n";

		write->w_ivector3({mapindexes[i].v1, mapindexes[i].v2, mapindexes[i].v3});
    }
	FS.w_close(write);

 //   file.close();
 //   Msg("Object data saved to: %s" , p_sdk);
}

void SaveToObject_CFORM(size_t v_count, Fvector* verts, size_t t_count, CDB::TRI* tris, char* filename)
{
	string_path p_sdk ;
	string256 ff;
	sprintf(ff, "worldobj\\cform\\%s",filename);

	FS.update_path(p_sdk, "$fs_root$", ff);

	Status("Save File: %s", ff);

 
	

	std::ofstream file(p_sdk);
    if (!file.is_open()) 
	{
        Msg("Unable to create file: %s", p_sdk );
        return;
    }

    // Записываем вершины
	for (int i = 0; i < v_count;i ++)
	{	
 
		if (i % 1024 == 0)
		StatusNoMSG("Size: %d/%d", i, v_count);
        file << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
    }

    // Записываем индексы 
	for (int i = 0; i < t_count; i ++) 
	{
		 
		if (i % 1024 == 0)
			StatusNoMSG("Size: %d/%d", i, t_count);
        file << "f " << tris[i].verts[0] + 1 << " " << tris[i].verts[1] + 1 << " " << tris[i].verts[2] + 1 << "\n";
    }

    file.close();
    Msg("Object data saved to: %s" , p_sdk);
}

void SaveToObject(size_t v_count, Fvector* verts, size_t t_count, std::vector<VertexMAP>& mapindexes, char* filename)
{
	string_path p_sdk ;
	string256 ff;
	sprintf(ff, "worldobj\\chunks\\%s",filename);

	FS.update_path(p_sdk, "$fs_root$", ff);

	Status("Save File: %s", ff);
	

	std::ofstream file(p_sdk);
    if (!file.is_open()) 
	{
        Msg("Unable to create file: %s", p_sdk );
        return;
    }

    // Записываем вершины
	for (int i = 0; i < v_count;i ++)
	{	
 
		if (i % 1024 == 0)
		StatusNoMSG("Size: %d/%d", i, v_count);
        file << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
    }

    // Записываем индексы 
	for (int i = 0; i < t_count; i ++) 
	{
		 
		if (i % 1024 == 0)
			StatusNoMSG("Size: %d/%d", i, t_count);
        file << "f " << mapindexes[i].v1 + 1 << " " << mapindexes[i].v2 + 1 << " " << mapindexes[i].v3 + 1 << "\n";
    }

    file.close();
    Msg("Object data saved to: %s" , p_sdk);
}


void SaveToGroup(std::ofstream& file, char* prefix, size_t v_count, Fvector* verts, size_t t_count, std::vector<VertexMAP>& mapindexes, int previosly_vertex)
{
	file << "g " << prefix << "\n";
	file << "\n";

    // Записываем вершины
	for (int i = 0; i < v_count;i ++)
	{	
 
		if (i % 1024 == 0)
		StatusNoMSG("Size: %d/%d", i, v_count);
        file << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
    }

    // Записываем индексы 
	for (int i = 0; i < t_count; i ++) 
	{
		 
		if (i % 1024 == 0)
			StatusNoMSG("Size: %d/%d", i, t_count);
        file << "f " << mapindexes[i].v1 + 1 + previosly_vertex << " " << mapindexes[i].v2 + 1 + previosly_vertex << " " << mapindexes[i].v3 + 1 + previosly_vertex << "\n";
    }
}

Fbox calculateAABB_for_triangle(Face& triangle) 
{
	Fvector v1 =  triangle.v[0]->P;
	Fvector v2 =  triangle.v[1]->P;
	Fvector v3 =  triangle.v[2]->P;

    Fbox aabb;
    aabb.min.x = std::min({v1.x, v2.x, v3.x});
    aabb.min.y = std::min({v1.y, v2.y, v3.y});
	aabb.min.z = std::min({v1.z, v2.z, v3.z});

	aabb.max.x = std::min({v1.x, v2.x, v3.x});
    aabb.max.y = std::min({v1.y, v2.y, v3.y});
	aabb.max.z = std::min({v1.z, v2.z, v3.z});
 
    return aabb;
}
 
void CollectFaces(CDB::CollectorPacked& CL)
{
	xr_vector<Face*>			adjacent_vec;
	adjacent_vec.reserve		(6*2*3); 

	for (vecFaceIt it=lc_global_data()->g_faces().begin(); it!=lc_global_data()->g_faces().end(); ++it)
	{
		Face*	F				= (*it);
		const Shader_xrLC&	SH		= F->Shader();
		if (!SH.flags.bLIGHT_CastShadow)					continue;

		Progress	(float(it-lc_global_data()->g_faces().begin())/float(lc_global_data()->g_faces().size()));
				
		// Collect
		adjacent_vec.clear	();
		for (int vit=0; vit<3; ++vit)
		{
			Vertex* V = F->v[vit];
			for (u32 adj=0; adj<V->m_adjacents.size(); adj++)
			{
				adjacent_vec.push_back(V->m_adjacents[adj]);
			}
		}

		std::sort		(adjacent_vec.begin(),adjacent_vec.end());
		adjacent_vec.erase	(std::unique(adjacent_vec.begin(),adjacent_vec.end()),adjacent_vec.end());

		// Unique
		BOOL			bAlready	= FALSE;

		for (u32 ait=0; ait<adjacent_vec.size(); ++ait)
		{
			Face*	Test					= adjacent_vec[ait];
			if (Test==F)					continue;
			if (!Test->flags.bProcessed)	continue;
			if (FaceEqual(*F,*Test))
			{
				bAlready					= TRUE;
				break;
			}
		}

		//
		if (!bAlready) 
		{
			F->flags.bProcessed	= true;
			CL.add_face_D		( F->v[0]->P,F->v[1]->P,F->v[2]->P, convert_nax(F), F->sm_group);
		}
 
	}
	 
}


int positiveMod(int a, int b) 
{
    return (a % b + b) % b; // Функция для получения положительного остатка от деления
}

/*
void CBuild::RunCollideFormNEW()
{
	CDB::CollectorPacked	CL	(scene_bb, lc_global_data()->g_vertices().size(), lc_global_data()->g_faces().size());
   
	CollectFaces(CL);
   
	vecVertex verts = lc_global_data()->g_vertices();
	vecFace faces = lc_global_data()->g_faces();


	Msg("Size Vertex[%u], Tris[%u]", verts.size(), faces.size() );
	
	Fbox RQBox;
	Fvector* vertsCL = CL.getV();
	for (int i = 0; i < CL.getVS(); i++)
	{
		RQBox.modify(vertsCL[i]);
 	}

	Msg("BBox: min[%f][%f][%f], max[%f][%f][%f]", VPUSH(RQBox.min), VPUSH(RQBox.max));
 
	int chunkSize = 128; // Размер чанка


	int worldX = abs(RQBox.min.x) + RQBox.max.x + (chunkSize * 2); // Предположим X мира
	int worldY = abs(RQBox.min.y) + RQBox.max.y; // Предположим Y мира
    int worldZ = abs(RQBox.min.z) + RQBox.max.z + (chunkSize * 2); // Предположим Z мира

	Msg("WorldZ: %d / min: %f / max: %f", worldZ, RQBox.min.z, RQBox.max.z);

	u32 numberOfChunksX = worldX / chunkSize;
	u32 numberOfChunksY = worldY / chunkSize;
    u32 numberOfChunksZ = worldZ / chunkSize;

	if (numberOfChunksY < 1)
		numberOfChunksY = 1;

	if (numberOfChunksX < 1)
		numberOfChunksX = 1;

	if (numberOfChunksZ < 1)
		numberOfChunksZ = 1;

	Msg("Calculate Chunks: %u / %u / %u", numberOfChunksX, numberOfChunksY, numberOfChunksZ);

	std::vector<std::vector<Face*>> chunks(numberOfChunksX * numberOfChunksY * numberOfChunksZ);
		
	Msg("Size Chunks: %d", chunks.size());


	

	for (int i = 0;i < faces.size();i++ )
	{
		Fbox box = calculateAABB_for_triangle(*faces[i]);
		
	
		// Определяем, в какой чанк попадает треугольник
        int chunkX = floorf(box.min.x / static_cast<float>(chunkSize)); //box.min.x / chunkSize;
	    int chunkY = floorf(box.min.y / static_cast<float>(chunkSize)); //box.min.y / chunkSize;
        int chunkZ = floorf(box.min.z / static_cast<float>(chunkSize));  //box.min.z / chunkSize;
		

 
		int chunkIndex =
				 positiveMod(chunkX, numberOfChunksX) +
                 positiveMod(chunkY, numberOfChunksY) * numberOfChunksX +
                 positiveMod(chunkZ, numberOfChunksZ) * (numberOfChunksX * numberOfChunksY);

		// Помещаем треугольник в соответствующий чанк

		if (chunkIndex < chunks.size() )
		{
			chunks[chunkIndex].push_back(faces[i]);
		}
		else 
		{
			Msg("Tri: %d", i);
			Msg("min.x: %u", (u32)box.min.x);
			Msg("min.z: %u", (u32)box.min.z);
			
			Msg("chunkX: %u", chunkX);
			Msg("chunkZ: %u", chunkZ);

			Msg("Save Chunk In Memory: %u / Max: %u", chunkZ * numberOfChunksX + chunkX, numberOfChunksX * numberOfChunksZ);
		}
	}
 
	Msg("Save Chunks");
 

#ifdef OTHER_FILES
	
	string_path p_sdk ;
	string256 ff;
	sprintf(ff, "worldobj\\%s", "chunked_world.obj");

	FS.update_path(p_sdk, "$fs_root$", ff);

	Status("Save File: %s", ff);
	
	std::ofstream file(p_sdk);
	  
	if (!file.is_open()) 
	{
        Msg("Unable to create file: %s", p_sdk );
        return;
    }

	int previosly_vertex = 0;

	for (int i = 0; i < chunks.size(); i++)
	{
		std::vector<Face*> triangles = chunks[i];
		std::vector<Fvector> vertexes;
	

		std::vector<VertexMAP> indexes(triangles.size());

		
		for (int tri_id = 0; tri_id < triangles.size(); tri_id++)
		{
			VertexMAP d;
			d.v1 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[0]->P);
			d.v2 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[1]->P);
			d.v3 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[2]->P);
			

			indexes[tri_id] = d;
		}

		
		string128 pref = {0};
		sprintf(pref, "Region_%d", i);
		
		
		if (vertexes.size() > 0)
		{
 			SaveToGroup(file, pref, vertexes.size(), vertexes.data(), triangles.size(), indexes, previosly_vertex);
 			file << "\n";
		}

		previosly_vertex += vertexes.size();
	}

	   
	file.close();
    Msg("Object data saved to: %s" , p_sdk);
#else 

	for (int i = 0; i < chunks.size(); i++)
	{
		std::vector<Face*> triangles = chunks[i];
		std::vector<Fvector> vertexes;
		std::vector<VertexMAP> indexes(triangles.size());

		
		for (int tri_id = 0; tri_id < triangles.size(); tri_id++)
		{
			VertexMAP d;
			d.v1 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[0]->P);
			d.v2 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[1]->P);
			d.v3 = vertexes.size(); 
			vertexes.push_back(triangles[tri_id]->v[2]->P);
			

			indexes[tri_id] = d;
		}

		
		string128 pref = {0};
		sprintf(pref, "Region_%d.obj", i);
		
 		SaveToBin(vertexes.size(), vertexes.data(), triangles.size(), indexes, pref);
  
	}
  

#endif
}
*/

void CBuild::Run(LPCSTR P)
{
	lc_global_data()->initialize();
#ifdef LOAD_GL_DATA
	net_light();
	return;
#endif

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);

	bool CformOnly = false;
	 
#pragma todo("se7kills TODO CFORM BUILD PARAMS ")

  	//****************************************** Open Level
	strconcat(sizeof(path), path, P, "\\");

	xr_strcpy(LEVEL_PATH, path);

	string_path					lfn;
	IWriter* fs = FS.w_open(strconcat(sizeof(lfn), lfn, path, "level."));
	fs->open_chunk(fsL_HEADER);
	hdrLEVEL H;
	H.XRLC_version = XRCL_PRODUCTION_VERSION;
	H.XRLC_quality = g_params().m_quality;
	fs->w(&H, sizeof(H));
	fs->close_chunk();
 
	//****************************************** Dumb entry in shader-registration
	RegisterShader("");

	//****************************************** Saving lights
	{
		string256			fn;
		IWriter* fs = FS.w_open(strconcat(sizeof(fn), fn, pBuild->path, "build.lights"));
		fs->w_chunk(0, &*L_static().rgb.begin(), L_static().rgb.size() * sizeof(R_Light));
		fs->w_chunk(1, &*L_static().hemi.begin(), L_static().hemi.size() * sizeof(R_Light));
		fs->w_chunk(2, &*L_static().sun.begin(), L_static().sun.size() * sizeof(R_Light));
		FS.w_close(fs);
	}
	 

	//****************************************** Optimizing + checking for T-junctions
	log_vminfo();
	
	FPU::m64r();
	Phase("Optimizing...");
	mem_Compact();
	if (!build_args->no_optimize)
		PreOptimize();
	CorrectTJunctions();
	
	log_vminfo();

	if (!CformOnly )
	{

	//****************************************** HEMI-Tesselate
		FPU::m64r();
		Phase("Adaptive HT...");
		mem_Compact();
		log_vminfo();
		xrPhase_AdaptiveHT();
	}

	log_vminfo();

	//****************************************** Collision DB
	//should be after normals, so that double-sided faces gets separated
 
	//****************************************** Building normals
	FPU::m64r();
	Phase("Building normals...");
	mem_Compact();
	CalcNormals();
	//SmoothVertColors			(5);

	FPU::m64r					();
	Phase						("Building collision database...");
	mem_Compact					();
	log_vminfo();
	BuildCForm					();
	log_vminfo();
	if (CformOnly)
		return;

	BuildPortals(*fs);

	//****************************************** T-Basis
	{
		FPU::m64r();
		Phase("Building tangent-basis...");
		xrPhase_TangentBasis();
		mem_Compact();
	}
 
 
	//****************************************** GLOBAL-RayCast model
	FPU::m64r();
	Phase("Building rcast-CFORM model...");
	mem_Compact();
	log_vminfo();
	Light_prepare();
	BuildRapid(TRUE);
	log_vminfo();
 


	//****************************************** GLOBAL-ILLUMINATION
	if (g_build_options.b_radiosity)			
	{
		FPU::m64r					();
		Phase						("Radiosity-Solver...");
		mem_Compact					();
		Light_prepare				();
		xrPhase_Radiosity			();
	}

	//****************************************** Starting MU
	/* 	Moved TO After LIGHT (После стадии Convert To OGF и возможность задать ключом -th потоки)	 (Возможно для сетевой компиляции стартуют раньше)
	FPU::m64r					();
	Phase						("LIGHT: Starting MU...");
	mem_Compact					();
	Light_prepare				();
	if(g_build_options.b_net_light)
	{
		lc_global_data()->mu_models_calc_materials();
		RunNetCompileDataPrepare( );
	}
	StartMu						();
	*/

	//****************************************** Resolve materials
	FPU::m64r					();
	Phase						("Resolving materials...");
	log_vminfo();
	mem_Compact					();
	xrPhase_ResolveMaterials	();
	IsolateVertices				(TRUE);

	log_vminfo();

	//****************************************** UV mapping
	{
		FPU::m64r					();
		Phase						("Build UV mapping...");
		
		mem_Compact					();
		log_vminfo();
		xrPhase_UVmap				();
		IsolateVertices				(TRUE);
	}
	
	log_vminfo();

	//****************************************** Subdivide geometry
	FPU::m64r					();
	Phase						("Subdividing geometry...");
	
	mem_Compact					();
	log_vminfo();
	xrPhase_Subdivide			();
	//IsolateVertices				(TRUE);
	lc_global_data()->vertices_isolate_and_pool_reload();

	log_vminfo();

	// Se7Kills
	// Export Model DEFLECTORS 
	// ExportDeflectors();


	// Se7Kills Opacity BUFFERS

	//****************************************** All lighting + lmaps building and saving
 		
	Light						();
	RunAfterLight				( fs );

}
 
void CBuild::	RunAfterLight			( IWriter* fs	)
{
 	//****************************************** Merge geometry
	FPU::m64r					();
	Phase						("Merging geometry...");
	mem_Compact					();
	xrPhase_MergeGeometry		();

	//****************************************** Convert to OGF
	FPU::m64r					();
	Phase						("Converting to OGFs...");
	mem_Compact					();
	Flex2OGF					();



	//****************************************** Export MU-models
	FPU::m64r					();
	Phase						("Converting MU-models to OGFs...");
	mem_Compact					();
	{
		u32 m;
		Status			("MU : Models...");
		for (m=0; m<mu_models().size(); m++)	
		{
			clMsg("ID[%d], size[%d]", m, mu_models().size());
			calc_ogf			(*mu_models()[m]);
			export_geometry		(*mu_models()[m]);
		}

		Status			("MU : References...");
		for (m = 0; m < mu_refs().size(); m++)
		{
			clMsg("muref ID[%d], size[%d]", m, mu_models().size());
			export_ogf(*mu_refs()[m]);
		}

//		lc_global_data()->clear_mu_models();
	}

	//****************************************** Destroy RCast-model
	FPU::m64r		();
	Phase			("Destroying ray-trace model...");
	mem_Compact		();
	lc_global_data()->destroy_rcmodel();
	
 
	//****************************************** Build sectors
	FPU::m64r();
	Phase("Building sectors...");
	mem_Compact();
	BuildSectors();
 
	//****************************************** Saving MISC stuff
	FPU::m64r		();
	Phase			("Saving...");
	mem_Compact		();
	SaveLights		(*fs);

	fs->open_chunk	(fsL_GLOWS);
	
	for (u32 i=0; i<glows.size(); i++)
	{
		b_glow&	G	= glows[i];
		fs->w		(&G,4*sizeof(float));
		string1024	sid;
		strconcat	(sizeof(sid),sid,
			shader_render[materials()[G.dwMaterial].shader].name,
			"/",
			textures()		[materials()[G.dwMaterial].surfidx].name
			);
		fs->w_u16	(RegisterShader(sid));
	}
	fs->close_chunk	();

	SaveTREE		(*fs);
	SaveSectors		(*fs);

	err_save		();
 
	mem_Compact();
}

void CBuild::err_save	()
{
	string_path		log_name;
	strconcat		(sizeof(log_name),log_name,"build_",Core.UserName,".err");
	FS.update_path	(log_name,"$logs$",log_name);

	IWriter*		fs	= FS.w_open(log_name);
	IWriter&		err = *fs;

	// t-junction
	err.open_chunk	(0);
	err.w_u32		(err_tjunction().size()/(1*sizeof(Fvector)));
	err.w			(err_tjunction().pointer(), err_tjunction().size());
	err.close_chunk	();

	// m-edje
	err.open_chunk	(1);
	err.w_u32		(err_multiedge().size()/(2*sizeof(Fvector)));
	err.w			(err_multiedge().pointer(), err_multiedge().size());
	err.close_chunk	();

	// invalid
	err.open_chunk	(2);
	err.w_u32		(err_invalid().size()/(3*sizeof(Fvector)));
	err.w			(err_invalid().pointer(), err_invalid().size());
	err.close_chunk	();

	FS.w_close( fs );
}

void CBuild::MU_ModelsCalculateNormals()
{
	for		(u32 m=0; m<mu_models().size(); m++)
		calc_normals( *mu_models()[m] );
}

xr_vector<xrMU_Model*>&CBuild::mu_models()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->mu_models(); 
}

xr_vector<xrMU_Reference*>&CBuild::mu_refs()
{
	VERIFY(lc_global_data()); 
	return lc_global_data()->mu_refs(); 
}

void CBuild::ImplicitLighting()
{
	::ImplicitLighting( g_build_options.b_net_light );
}