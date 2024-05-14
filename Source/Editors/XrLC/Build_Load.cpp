#include "stdafx.h"
#pragma optimize( "", off )
#include "elight_def.h"



#include "build.h"

#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"


#include "../xrLCLight/xrMU_Model.h"
#include "../xrLCLight/xrMU_Model_Reference.h"

// #define STB_IMAGE_IMPLEMENTATION
// #include "StbImage\stb_image.h"
#include "../DirectXTex/DirectXTex.h"
#pragma comment(lib, "DirectXTex.lib")


std::string DXGI_Get(DXGI_FORMAT dxgi)
{
	const char* formatStrings[117] =
	{
	 	"DXGI_FORMAT_UNKNOWN",
	 	"DXGI_FORMAT_R32G32B32A32_TYPELESS",
	 	"DXGI_FORMAT_R32G32B32A32_FLOAT",
	 	"DXGI_FORMAT_R32G32B32A32_UINT",
	 	"DXGI_FORMAT_R32G32B32A32_SINT",
	 	"DXGI_FORMAT_R32G32B32_TYPELESS",
	 	"DXGI_FORMAT_R32G32B32_FLOAT",
	 	"DXGI_FORMAT_R32G32B32_UINT",
	 	"DXGI_FORMAT_R32G32B32_SINT",
	 	"DXGI_FORMAT_R16G16B16A16_TYPELESS",
	 	"DXGI_FORMAT_R16G16B16A16_FLOAT",
	 	"DXGI_FORMAT_R16G16B16A16_UNORM",
	 	"DXGI_FORMAT_R16G16B16A16_UINT",
	 	"DXGI_FORMAT_R16G16B16A16_SNORM",
	 	"DXGI_FORMAT_R16G16B16A16_SINT",
	 	"DXGI_FORMAT_R32G32_TYPELESS",
	 	"DXGI_FORMAT_R32G32_FLOAT",
	 	"DXGI_FORMAT_R32G32_UINT",
	 	"DXGI_FORMAT_R32G32_SINT",
	 	"DXGI_FORMAT_R32G8X24_TYPELESS",
	 	"DXGI_FORMAT_D32_FLOAT_S8X24_UINT",
	 	"DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS",
	 	"DXGI_FORMAT_X32_TYPELESS_G8X24_UINT",
	 	"DXGI_FORMAT_R10G10B10A2_TYPELESS",
	 	"DXGI_FORMAT_R10G10B10A2_UNORM",
	 	"DXGI_FORMAT_R10G10B10A2_UINT",
	 	"DXGI_FORMAT_R11G11B10_FLOAT",
	 	"DXGI_FORMAT_R11G11B10_FLOAT",
	 	"DXGI_FORMAT_R8G8B8A8_UNORM",
	 	"DXGI_FORMAT_R8G8B8A8_UNORM_SRGB",
	 	"DXGI_FORMAT_R8G8B8A8_UINT",
	 	"DXGI_FORMAT_R8G8B8A8_SNORM",
	 	"DXGI_FORMAT_R8G8B8A8_SINT",
	 	"DXGI_FORMAT_R16G16_TYPELESS",
	 	"DXGI_FORMAT_R16G16_FLOAT",
	 	"DXGI_FORMAT_R16G16_UNORM",
	 	"DXGI_FORMAT_R16G16_UINT",
	 	"DXGI_FORMAT_R16G16_SNORM",
	 	"DXGI_FORMAT_R16G16_SINT",
	 	"DXGI_FORMAT_R32_TYPELESS",
	 	"DXGI_FORMAT_D32_FLOAT",
	 	"DXGI_FORMAT_R32_FLOAT",
	 	"DXGI_FORMAT_R32_UINT",
	 	"DXGI_FORMAT_R32_SINT",
	 	"DXGI_FORMAT_R24G8_TYPELESS",
	 	"DXGI_FORMAT_D24_UNORM_S8_UINT",
	 	"DXGI_FORMAT_R24_UNORM_X8_TYPELESS",
	 	"DXGI_FORMAT_X24_TYPELESS_G8_UINT",
	 	"DXGI_FORMAT_R8G8_TYPELESS",
	 	"DXGI_FORMAT_R8G8_UNORM",
	 	"DXGI_FORMAT_R8G8_UINT",
	 	"DXGI_FORMAT_R8G8_SNORM",
	 	"DXGI_FORMAT_R8G8_SINT",
	 	"DXGI_FORMAT_R16_TYPELESS",
	 	"DXGI_FORMAT_R16_FLOAT",
	 	"DXGI_FORMAT_D16_UNORM",
	 	"DXGI_FORMAT_R16_UNORM",
	 	"DXGI_FORMAT_R16_UINT",
	 	"DXGI_FORMAT_R16_SNORM",
	 	"DXGI_FORMAT_R16_SINT",
	 	"DXGI_FORMAT_R8_TYPELESS",
	 	"DXGI_FORMAT_R8_UNORM",
	 	"DXGI_FORMAT_R8_UINT",
	 	"DXGI_FORMAT_R8_SNORM",
	 	"DXGI_FORMAT_R8_SINT",
	 	"DXGI_FORMAT_A8_UNORM",
	 	"DXGI_FORMAT_R1_UNORM",
	 	"DXGI_FORMAT_R9G9B9E5_SHAREDEXP",
	 	"DXGI_FORMAT_R8G8_B8G8_UNORM",
	 	"DXGI_FORMAT_G8R8_G8B8_UNORM",
	 	"DXGI_FORMAT_BC1_TYPELESS",
	 	"DXGI_FORMAT_BC1_UNORM",
	 	"DXGI_FORMAT_BC1_UNORM_SRGB",
	 	"DXGI_FORMAT_BC2_TYPELESS",
	 	"DXGI_FORMAT_BC2_UNORM",
	 	"DXGI_FORMAT_BC2_UNORM_SRGB",
	 	"DXGI_FORMAT_BC3_TYPELESS",
	 	"DXGI_FORMAT_BC3_UNORM",
	 	"DXGI_FORMAT_BC3_UNORM_SRGB",
	 	"DXGI_FORMAT_BC4_TYPELESS",
	 	"DXGI_FORMAT_BC4_UNORM",
	 	"DXGI_FORMAT_BC4_SNORM",
	 	"DXGI_FORMAT_BC5_TYPELESS",
	 	"DXGI_FORMAT_BC5_UNORM",
	 	"DXGI_FORMAT_BC5_SNORM",
	 	"DXGI_FORMAT_B5G6R5_UNORM",
	 	"DXGI_FORMAT_B5G5R5A1_UNORM",
	 	"DXGI_FORMAT_B8G8R8A8_UNORM",
	 	"DXGI_FORMAT_B8G8R8X8_UNORM",
	 	"DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM",
	 	"DXGI_FORMAT_B8G8R8A8_TYPELESS",
	 	"DXGI_FORMAT_B8G8R8A8_UNORM_SRGB",
	 	"DXGI_FORMAT_B8G8R8X8_TYPELESS",
	 	"DXGI_FORMAT_B8G8R8X8_UNORM_SRGB",
	 	"DXGI_FORMAT_BC6H_TYPELESS",
	 	"DXGI_FORMAT_BC6H_UF16",
	 	"DXGI_FORMAT_BC6H_SF16",
	 	"DXGI_FORMAT_BC7_TYPELESS",
	 	"DXGI_FORMAT_BC7_UNORM",
	 	"DXGI_FORMAT_BC7_UNORM_SRGB",
	 	"DXGI_FORMAT_AYUV",
	 	"DXGI_FORMAT_Y410",
	 	"DXGI_FORMAT_Y416",
	 	"DXGI_FORMAT_NV12",
	 	"DXGI_FORMAT_P010",
	 	"DXGI_FORMAT_P016",
	 	"DXGI_FORMAT_420_OPAQUE",
	 	"DXGI_FORMAT_YUY2",
	 	"DXGI_FORMAT_Y210",
	 	"DXGI_FORMAT_Y210",
	 	"DXGI_FORMAT_NV11",
	 	"DXGI_FORMAT_AI44",
	 	"DXGI_FORMAT_IA44",
	 	"DXGI_FORMAT_P8",
	 	"DXGI_FORMAT_A8P8",
	 	"DXGI_FORMAT_A8P8",
		"DXGI_FORMAT_B4G4R4A4_UNORM"
 
 
	};

	if (dxgi < 117)
	{
		return formatStrings[dxgi];
	}
	else
		return "Undefined DXT Format";
}


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

#include <windows.h>
#include <string>

std::wstring ConvertToWideString(const std::string& narrowString)
{
	int wideStrLength = MultiByteToWideChar(CP_UTF8, 0, narrowString.c_str(), -1, nullptr, 0);
	if (wideStrLength == 0)
	{
		// ��������� ������ �����������
		return L"not converted";
	}

	std::wstring wideString;
	wideString.resize(wideStrLength);

	if (MultiByteToWideChar(CP_UTF8, 0, narrowString.c_str(), -1, &wideString[0], wideStrLength) == 0)
	{
		// ��������� ������ �����������
		return L"not converted";
	}

	return wideString;
}


std::string ConvertToString(const std::wstring& narrowString)
{
	int wideStrLength = WideCharToMultiByte(CP_UTF8, 0, narrowString.c_str(), -1, nullptr, 0, nullptr, nullptr);
	if (wideStrLength == 0)
	{
		// ��������� ������ �����������
		return "not converted";
	}

	std::string wideString;
	wideString.resize(wideStrLength);

	if (WideCharToMultiByte(CP_UTF8, 0, narrowString.c_str(), -1, &wideString[0], wideStrLength, nullptr, nullptr) == 0)
	{
		// ��������� ������ �����������
		return "not converted";
	}

	return wideString;
}


int DXTCompressImageXRLC(LPCSTR out_name, u32* raw_data, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth)
{
	Msg("DXT: Compressing Image: %s %uX%u", out_name, w, h);

 	DirectX::TexMetadata meta;
	DirectX::ScratchImage sqImage;


	size_t dataSize = w * h * sizeof(u32);

	HRESULT HK = DirectX::LoadFromDDSMemory(raw_data, dataSize, DirectX::DDS_FLAGS_NONE, &meta, sqImage);
	
	Msg("HK: 0x%p", HK);

	if (HK == 0)
	clMsg("[ConvertDXT] Reading DDS: Images: %d Width: %u, Height: %u, \\n Metadata W: %u, H: %u Pixels: %u, Format: %d, %s, AlphaMode: %d",
		sqImage.GetImageCount(),
		sqImage.GetImages()->width,
		sqImage.GetImages()->height,
		sqImage.GetMetadata().width,
		sqImage.GetMetadata().height,
		sqImage.GetPixelsSize() / 4,
		sqImage.GetMetadata().format,
		DXGI_Get(sqImage.GetMetadata().format).c_str(),
		sqImage.GetMetadata().GetAlphaMode());



	//DirectX::SaveToDDSFile(sqImage.GetImages()[0], DirectX::DDS_FLAGS_NONE, ConvertToWideString(out_name).c_str());


	Msg("Loaded : TRUE");
 

	Msg("DXT: Compressing Image: 2 [Closing File]");
	return  1;
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
 
inline bool Surface_Export(string_path& F, LPSTR N)
{
	FS.update_path(F, "$game_tex_exports_export$", strconcat(sizeof(F), F, N, ".dds"));
	IWriter* w = FS.w_open(F);
	FS.w_close(w);

	return true;
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
			} catch (...)
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
				F->r		(&temp,sizeof(temp));
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
			if (0==xr_strcmp(N,"level_lods"))	{
				// HACK for merged lod textures
				BT.dwWidth		= 1024;
				BT.dwHeight		= 1024;
				BT.bHasAlpha	= TRUE;
				BT.THM.SetHasSurface(FALSE);
				BT.pSurface = 0;

			} else {
				string_path			th_name;
				FS.update_path	(th_name,"$game_textures$",strconcat(sizeof(th_name),th_name,N,".thm"));
				clMsg			("processing: %s",th_name);
				IReader* THM	= FS.r_open(th_name);
				R_ASSERT2		(THM,th_name);

				// version
				u32 version = 0;
				R_ASSERT2(THM->r_chunk(THM_CHUNK_VERSION,&version),th_name);
				// if( version!=THM_CURRENT_VERSION )	FATAL	("Unsupported version of THM file.");

				// analyze thumbnail information
				R_ASSERT2(THM->find_chunk(THM_CHUNK_TEXTUREPARAM),th_name);
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
				BT.dwWidth	= BT.THM.width;
				BT.dwHeight	= BT.THM.height;
				BT.bHasAlpha= BT.THM.HasAlphaChannel();

				if (!bLOD) 
				{
					if (BT.bHasAlpha || BT.THM.flags.test(STextureParams::flImplicitLighted) || g_build_options.b_radiosity)
					{
					
						string_path name;
						R_ASSERT( Surface_Detect(name, N) );
 
						BT.pSurface = 0;
						BT.THM.SetHasSurface(true);
						   
					 

						 
						std::wstring& str = ConvertToWideString(name);	

 
						DirectX::TexMetadata metadata;
						DirectX::ScratchImage sqImage;

						DirectX::LoadFromDDSFile(str.c_str(), DirectX::DDS_FLAGS::DDS_FLAGS_NONE, &metadata, sqImage);
						clMsg("Reading DDS: Images: %d Width: %u, Height: %u, \\n Metadata W: %u, H: %u Pixels: %u, Format: %d, %s, AlphaMode: %d",
							sqImage.GetImageCount(),
							sqImage.GetImages()->width,
							sqImage.GetImages()->height,
							sqImage.GetMetadata().width,
							sqImage.GetMetadata().height,
							sqImage.GetPixelsSize() / 4,
							sqImage.GetMetadata().format,
							DXGI_Get(sqImage.GetMetadata().format).c_str(),
							sqImage.GetMetadata().GetAlphaMode());



						DirectX::ScratchImage decodeImage;

						HRESULT hk = DirectX::Decompress( sqImage.GetImages(), sqImage.GetImageCount(), sqImage.GetMetadata(), DXGI_FORMAT_R8G8B8A8_UNORM, decodeImage);
						
						u32* raw_data = xr_alloc<u32>(decodeImage.GetMetadata().width * decodeImage.GetMetadata().height);

						//if (hk != 0 )
						{
							switch (hk)
							{
							case S_OK:
							{
								clMsg("Decompressed DDS: Images: %d Width: %u, Height: %u, \\n Metadata W: %u, H: %u Pixels: %u, Format: %d, %s, AlphaMode: %d",
									decodeImage.GetImageCount(),
									decodeImage.GetImages()->width,
									decodeImage.GetImages()->height,
									decodeImage.GetMetadata().width,
									decodeImage.GetMetadata().height,
									decodeImage.GetPixelsSize() / 4,
									decodeImage.GetMetadata().format,
									DXGI_Get(decodeImage.GetMetadata().format).c_str(),
									decodeImage.GetMetadata().GetAlphaMode());

								clMsg("�������� ��������� �������.");

								for (auto ImageID = 0; ImageID < sqImage.GetImageCount(); ImageID++)
								{
									Msg("Decompressed DDS: For Image[%d]: pixels: %u",
										ImageID, sqImage.GetImages()[ImageID].pixels);

								//	Msg("Decompressed DDS: Width: %llu, Height: %llu", 
								//		sqImage.GetImages()[ImageID].width, sqImage.GetImages()[ImageID].height);
								}
								 

								u32* pixels = reinterpret_cast<u32*>( decodeImage.GetPixels() );
								int width = decodeImage.GetMetadata().width;
								int height = decodeImage.GetMetadata().height;
								
								int RequriedPixels = width * height * 4;

								if (decodeImage.GetPixelsSize() < RequriedPixels)
									Msg(" Cant Read Pixels : Width * height (%u) < %u", RequriedPixels, decodeImage.GetPixelsSize());
								else
								{
									Msg(" ������ ����� � ����� �������� !!!! (%u) < %u", RequriedPixels, decodeImage.GetPixelsSize());


									for (size_t pixel = 0; pixel < width * height; ++pixel)
									{
										 
										// ���������� ������� ������� � ������� ������
										 

										if (pixel < decodeImage.GetPixelsSize())
										{
											raw_data[pixel] = pixels[pixel];
										}
										else
										{
											Msg("Pixel Index Is Bad: %u", pixel);
										}
										

 										// ������ � ��� ���� �������� ������� RGB � A ��� �������� �������.
										// �� ������ ������������ ��� �������� ��� �������� ��������� ��� ������ ��������.
										 
									}
								}
								

							}break;

							case E_FAIL:
							{
								clMsg("����� ������, �������� �� �������.");
							}break;

							case E_INVALIDARG:
							{
								clMsg("������������ ��������.");
							}break;

							case E_OUTOFMEMORY:
							{
								clMsg("������������ ������ ��� ���������� ��������.");
							}break;

							case E_NOTIMPL:
							{
								clMsg("����� ��� ������� �� �����������.");
							}break;

							case E_POINTER:
							{
								clMsg("������������ ���������.");

							}break;

							case E_ACCESSDENIED:
							{
								clMsg("������ ��������.");
							}break;
 
							default:
								clMsg("Reason: %u", hk);
								break;
							}

 						}
						
			 
						if (S_OK == hk)
						{
							BT.pSurface = raw_data;
 
							string_path path;
							Surface_Export(path, N);

							DXTCompressImageXRLC(path, raw_data, decodeImage.GetMetadata().width, decodeImage.GetMetadata().height, 0, &BT.THM, 4);
  
							BT.directXPixelsSize = sqImage.GetPixelsSize() / 4;

							if ((sqImage.GetImages()->width != BT.dwWidth) || (sqImage.GetImages()->height != BT.dwHeight))
							{
								clMsg("! THM doesn't correspond to the texture: %dx%d -> %dx%d",
									BT.dwWidth, BT.dwHeight,
									sqImage.GetImages()->width,
									sqImage.GetImages()->height
								);

								BT.dwWidth = BT.THM.width = sqImage.GetImages()->width;
								BT.dwHeight = BT.THM.height = sqImage.GetImages()->height;
							}
						}
						else
						{
							BT.THM.SetHasSurface(FALSE);
						}
 

						
 
					} 
					else
					{
						// Free surface memory
					}
				}
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

	// 
	clMsg	("* sizes: V(%d),F(%d)",sizeof(Vertex),sizeof(Face));
}



