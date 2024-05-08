#pragma once

#include "jsonxx/jsonxx.h"
using namespace jsonxx;

class UIDxtConverter :  public XrUI
{
	static UIDxtConverter* Form_DXT;	
	struct THM
	{
		// NAME FILE
		shared_str name_dds;

		STextureParams::ETFormat fmt;
		Flags32 flags;
		u32 border_color;
		u32 fade_color;
		u32 fade_amount;
		u32 width;
		u32 height;
		u32 mip_filter;

		// chunk type
		u32 type;

		//chunk Detail Ext
		shared_str detail_name;
		float detail_scale;

		// chunk material 
		u32 material;
		float material_weight;

		// chunk bump
		float bump_virtual_height;
		u32 bump_mode;
		shared_str bump_name;

		// chunk NormalMap
		shared_str ext_normal_map_name;

		// Chunk fade
		u8 fade_delay;


		/*
			flGenerateMipMaps	= (1<<0),
			flBinaryAlpha		= (1<<1),
			flAlphaBorder		= (1<<4),
			flColorBorder		= (1<<5),
			flFadeToColor		= (1<<6),
			flFadeToAlpha		= (1<<7),
			flDitherColor		= (1<<8),
			flDitherEachMIPLevel= (1<<9),

			flDiffuseDetail		= (1<<23),
			flImplicitLighted	= (1<<24),
			flHasAlpha			= (1<<25),
			flBumpDetail		= (1<<26),

		*/


		LPCSTR CvrtFormat()
		{
			switch (fmt)
			{
			case STextureParams::tfDXT1:
				return "tfDXT1";
				break;
			case STextureParams::tfADXT1:
				return "tfADXT1";
				break;
			case STextureParams::tfDXT3:
				return "tfDXT3";
				break;
			case STextureParams::tfDXT5:
				return "tfDXT5";
				break;
			case STextureParams::tf4444:
				return "tf4444";
				break;
			case STextureParams::tf1555:
				return "tf1555";
				break;
			case STextureParams::tf565:
				return "tf565";
				break;
			case STextureParams::tfRGB:
				return "tfRGB";
				break;
			case STextureParams::tfRGBA:
				return "tfRGBA";
				break;
			case STextureParams::tfNVHS:
				return "tfNVHS";
				break;
			case STextureParams::tfNVHU:
				return "tfNVHU";
				break;
			case STextureParams::tfA8:
				return "tfA8";
				break;
			case STextureParams::tfL8:
				return "tfL8";
				break;
			case STextureParams::tfA8L8:
				return "tfA8L8";
				break;
			case STextureParams::tfBC4:
				return "tfBC4";
				break;
			case STextureParams::tfBC5:
				return "tfBC5";
				break;
			case STextureParams::tfBC6:
				return "tfBC6";
				break;
			case STextureParams::tfBC7:
				return "tfBC7";
				break;
			case STextureParams::tfForceU32:
				return "tfForceU32";
				break;
			default:
				break;
			}
		}

		STextureParams::ETFormat CvrtFormatBack(LPCSTR name)
		{
			if (strstr(name, "tfDXT1"))
				return STextureParams::tfDXT1;

			if (strstr(name, "tfADXT1"))
				return STextureParams::tfADXT1;

			if (strstr(name, "tfDXT3"))
				return STextureParams::tfDXT3;

			if (strstr(name, "tfDXT5"))
				return STextureParams::tfDXT5;

			if (strstr(name, "tf4444"))
				return STextureParams::tf4444;

			if (strstr(name, "tf1555"))
				return STextureParams::tf1555;

			if (strstr(name, "tf565"))
				return STextureParams::tf565;

			if (strstr(name, "tfRGB"))
				return STextureParams::tfRGB;

			if (strstr(name, "tfRGBA"))
				return STextureParams::tfRGBA;

			if (strstr(name, "tfNVHS"))
				return STextureParams::tfNVHS;

			if (strstr(name, "tfNVHU"))
				return STextureParams::tfNVHU;

			if (strstr(name, "tfA8"))
				return STextureParams::tfA8;

			if (strstr(name, "tfL8"))
				return STextureParams::tfL8;

			if (strstr(name, "tfBC4"))
				return STextureParams::tfBC4;

			if (strstr(name, "tfBC5"))
				return STextureParams::tfBC5;

			if (strstr(name, "tfBC6"))
				return STextureParams::tfBC6;

			if (strstr(name, "tfBC7"))
				return STextureParams::tfBC7;

			return STextureParams::tfForceU32;
		}

		void save_thm(CInifile* file)
		{
			LPCSTR section = name_dds.c_str();

			file->w_string(section, "fmt", CvrtFormat());

			file->w_u32(section, "flags", flags.get());
			file->w_u32(section, "border_color", border_color);
			file->w_u32(section, "fade_color", fade_color);
			file->w_u32(section, "fade_amount", fade_amount);
			file->w_u32(section, "width", width);
			file->w_u32(section, "height", height);
			file->w_u32(section, "mip_filter", mip_filter);

			file->w_bool(section, "flGenerateMipMaps", flags.test(STextureParams::flGenerateMipMaps) );
			file->w_bool(section, "flBinaryAlpha", flags.test(STextureParams::flBinaryAlpha));
			file->w_bool(section, "flColorBorder", flags.test(STextureParams::flColorBorder));
			file->w_bool(section, "flFadeToColor", flags.test(STextureParams::flFadeToColor));
			file->w_bool(section, "flFadeToAlpha", flags.test(STextureParams::flFadeToAlpha));
			file->w_bool(section, "flDitherColor", flags.test(STextureParams::flDitherColor));
			file->w_bool(section, "flDitherEachMIPLevel", flags.test(STextureParams::flDitherEachMIPLevel));
			file->w_bool(section, "flDiffuseDetail", flags.test(STextureParams::flDiffuseDetail));
			file->w_bool(section, "flImplicitLighted", flags.test(STextureParams::flImplicitLighted));
			file->w_bool(section, "flHasAlpha", flags.test(STextureParams::flHasAlpha));
			file->w_bool(section, "flBumpDetail", flags.test(STextureParams::flBumpDetail));
 


			//
			file->w_u32(section, "type", type);

			//
			file->w_string(section, "detail_name", detail_name.c_str());
			file->w_float(section, "detail_scale", detail_scale);

			//
			file->w_u32(section, "material", material);
			file->w_float(section, "material_weight", material_weight);

			//
			file->w_float(section, "bump_virtual_height", bump_virtual_height);
			file->w_u32(section, "bump_mode", bump_mode);
			file->w_string(section, "bump_name", bump_name.c_str());

			// 
			file->w_string(section, "ext_normal_map_name", ext_normal_map_name.c_str());

			//
			file->w_u8(section, "fade_delay", fade_delay);
		}

		void load_thm(CInifile* file, LPCSTR section)
		{
			name_dds = section;

			fmt = CvrtFormatBack(file->r_string(section, "fmt"));

			flags.flags = file->r_u32(section, "flags");
			border_color = file->r_u32(section, "border_color");
			fade_color = file->r_u32(section, "fade_color");
			fade_amount = file->r_u32(section, "fade_amount");
			width = file->r_u32(section, "width");
			height = file->r_u32(section, "height");
			mip_filter = file->r_u32(section, "mip_filter");

			//
			type = file->r_u32(section, "type");

			//
			detail_name = file->r_string(section, "detail_name");
			detail_scale = file->r_float(section, "detail_scale");

			//
			material = file->r_u32(section, "material");
			material_weight = file->r_float(section, "material_weight");

			//
			bump_virtual_height = file->r_float(section, "bump_virtual_height");
			bump_mode = file->r_u32(section, "bump_mode");
			bump_name = file->r_string(section, "bump_name");

			// 
			ext_normal_map_name = file->r_string(section, "ext_normal_map_name");

			//
			fade_delay = file->r_u8(section, "fade_delay");
		}

		void ReadFromReader(IReader* F, LPCSTR name)
		{
			name_dds = name;

			if (F->open_chunk(THM_CHUNK_TEXTUREPARAM))
			{
				F->r(&fmt, sizeof(STextureParams::ETFormat));
				flags.flags = F->r_u32();
				border_color = F->r_u32();
				fade_color = F->r_u32();
				fade_amount = F->r_u32();
				mip_filter = F->r_u32();
				width = F->r_u32();
				height = F->r_u32();
			}

			if (F->open_chunk(THM_CHUNK_TEXTURE_TYPE))
				type = F->r_u32();

			if (F->open_chunk(THM_CHUNK_DETAIL_EXT))
			{
				F->r_stringZ(detail_name);
				detail_scale = F->r_float();
			}

			if (F->open_chunk(THM_CHUNK_MATERIAL))
			{
				material = F->r_u32();
				material_weight = F->r_float();
			}

			if (F->open_chunk(THM_CHUNK_BUMP))
			{
				bump_virtual_height = F->r_float();
				bump_mode = F->r_u32();
				F->r_stringZ(bump_name);
			}

			if (F->open_chunk(THM_CHUNK_EXT_NORMALMAP))
			{
				F->r_stringZ(ext_normal_map_name);
			}

			if (F->open_chunk(THM_CHUNK_FADE_DELAY))
			{
				fade_delay = F->r_u8();
			}

		}
	
		jsonxx::Object save_json()
		{
			jsonxx::Object thm_json;
			thm_json << "fmt" << String(CvrtFormat());

			thm_json << "flags" << Number(flags.get());
			thm_json << "border_color" << Number(border_color);

			thm_json << "fade_color" << Number(fade_color);
			thm_json << "fade_amount" << Number(fade_amount);
			thm_json << "width" << Number(width);
			thm_json << "height" << Number(height);
			thm_json << "mip_filter" << Number(mip_filter);
			
			thm_json << "flGenerateMipMaps" << Number(flags.test(STextureParams::flGenerateMipMaps));
			thm_json << "flBinaryAlpha" << Number(flags.test(STextureParams::flBinaryAlpha));
			thm_json << "flColorBorder" << Number(flags.test(STextureParams::flColorBorder));
			thm_json << "flFadeToColor" << Number(flags.test(STextureParams::flFadeToColor));
			thm_json << "flFadeToAlpha" << Number(flags.test(STextureParams::flFadeToAlpha));
			thm_json << "flDitherColor" << Number(flags.test(STextureParams::flDitherColor));
			thm_json << "flDitherEachMIPLevel" << Number(flags.test(STextureParams::flDitherEachMIPLevel));
 			thm_json << "flDiffuseDetail" << Number(flags.test(STextureParams::flDiffuseDetail));
			thm_json << "flImplicitLighted" << Number(flags.test(STextureParams::flImplicitLighted));
			thm_json << "flHasAlpha" << Number(flags.test(STextureParams::flHasAlpha));
 			thm_json << "flBumpDetail" << Number(flags.test(STextureParams::flBumpDetail));

			thm_json << "type" << Number(type);
			thm_json << "detail_name" << String(detail_name.c_str());
			thm_json << "detail_scale" << Number(detail_scale);

			thm_json << "material" << Number(material);
			thm_json << "material_weight" << Number(material_weight);
 
			thm_json << "bump_virtual_height" << Number(bump_virtual_height);
			thm_json << "bump_mode" << Number(bump_mode);
			thm_json << "bump_name" << String(bump_name.c_str());
	
			thm_json << "ext_normal_map_name" << String(ext_normal_map_name.c_str());
			thm_json << "fade_delay" << Number(flags.test(STextureParams::flBumpDetail));


			LPCSTR section = name_dds.c_str();
			//array_json << section << thm_json;

			return thm_json;
		
		}
};



private: 
	virtual void Draw() override;
public:
	static void Update();
	static void Show();
	static void Close();

 
	
	static IC bool IsOpen()  { return Form_DXT; }
};

