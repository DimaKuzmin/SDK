// DXT.cpp : Defines the entry point for the DLL application.
//

#include "stdafx.h"
#include "../../BearBundle/External/Public/nvtt/nvtt/nvtt.h"

class DDSErrorHandler : public nvtt::ErrorHandler
{
public:
    virtual void error(nvtt::Error e) override;
};

void DDSErrorHandler::error(nvtt::Error e)
{
    Msg("Error Export: %s", nvtt::errorString(e));
   // MessageBox(0, nvtt::errorString(e), "DXT compress error", MB_ICONERROR | MB_OK);
}

u32* Build32MipLevel(u32& _w, u32& _h, u32& _p, u32* pdwPixelSrc, STextureParams* fmt, float blend)
{
    R_ASSERT(pdwPixelSrc);
    R_ASSERT(_w % 2 == 0);
    R_ASSERT(_h % 2 == 0);
    R_ASSERT(_p % 4 == 0);
    u32 dwDestPitch = (_w / 2) * 4;
    u32* pNewData = xr_alloc<u32>((_h / 2) * dwDestPitch);
    u8* pDest = (u8*)pNewData;
    u8* pSrc = (u8*)pdwPixelSrc;
    float mixed_a = (float)u8(fmt->fade_color >> 24);
    float mixed_r = (float)u8(fmt->fade_color >> 16);
    float mixed_g = (float)u8(fmt->fade_color >> 8);
    float mixed_b = (float)u8(fmt->fade_color >> 0);
    float inv_blend = 1.f - blend;
    for (u32 y = 0; y < _h; y += 2)
    {
        u8* pScanline = pSrc + y * _p;
        for (u32 x = 0; x < _w; x += 2)
        {
            u8* p1 = pScanline + x * 4;
            u8* p2 = p1 + 4;
            if (1 == _w)
                p2 = p1;
            u8* p3 = p1 + _p;
            if (1 == _h)
                p3 = p1;
            u8* p4 = p2 + _p;
            if (1 == _h)
                p4 = p2;
            float c_r = float(u32(p1[0]) + u32(p2[0]) + u32(p3[0]) + u32(p4[0])) / 4.f;
            float c_g = float(u32(p1[1]) + u32(p2[1]) + u32(p3[1]) + u32(p4[1])) / 4.f;
            float c_b = float(u32(p1[2]) + u32(p2[2]) + u32(p3[2]) + u32(p4[2])) / 4.f;
            float c_a = float(u32(p1[3]) + u32(p2[3]) + u32(p3[3]) + u32(p4[3])) / 4.f;
            if (fmt->flags.is(STextureParams::flFadeToColor))
            {
                c_r = c_r * inv_blend + mixed_r * blend;
                c_g = c_g * inv_blend + mixed_g * blend;
                c_b = c_b * inv_blend + mixed_b * blend;
            }
            if (fmt->flags.is(STextureParams::flFadeToAlpha))
            {
                c_a = c_a * inv_blend + mixed_a * blend;
            }
            float A = c_a + c_a / 8.f;
            int _r = int(c_r);
            clamp(_r, 0, 255);
            *pDest++ = u8(_r);
            int _g = int(c_g);
            clamp(_g, 0, 255);
            *pDest++ = u8(_g);
            int _b = int(c_b);
            clamp(_b, 0, 255);
            *pDest++ = u8(_b);
            int _a = int(A);
            clamp(_a, 0, 255);
            *pDest++ = u8(_a);
        }
    }
    _w /= 2;
    _h /= 2;
    _p = _w * 4;
    return pNewData;
}

void FillRect(u8* data, u8* new_data, u32 offs, u32 pitch, u32 h, u32 full_pitch)
{
    for (u32 i = 0; i < h; i++)
    {
        CopyMemory(data + (full_pitch * i + offs), new_data + i * pitch, pitch);
    }
}

IC u32 GetPowerOf2Plus1(u32 v)
{
    u32 cnt = 0;
    while (v)
    {
        v >>= 1;
        cnt++;
    }
    return cnt;
}

#include "DirectXTex/DirectXTex.h"
#pragma comment(lib, "DirectXTex.lib")
#pragma optimize(off, "")

int DXTCompressImage	(LPCSTR out_name, u8* raw_data, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth)
{
   /*
     
    DirectX::ScratchImage image;
    DirectX::ScratchImage result;
    

    DirectX::TexMetadata metadata;
    metadata.width = w;
    metadata.height = h;
 
    DXGI_FORMAT Format;
     
    switch (fmt->fmt)
    {
    case STextureParams::tfADXT1:
    case STextureParams::tfDXT1:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC1_UNORM;
        break;

    case STextureParams::tfDXT3:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC2_UNORM;
        break;

    case STextureParams::tfDXT5:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC3_UNORM;
        break;

    case STextureParams::tfBC4:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC4_UNORM;
        break;

    case STextureParams::tfBC5:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC5_UNORM;
        break;

    case STextureParams::tfBC6:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC6H_SF16;
        break;

    case STextureParams::tfBC7:
        Format = DXGI_FORMAT::DXGI_FORMAT_BC7_UNORM;
        break;

    case STextureParams::tfRGB:
        Format = DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
        break;

    case STextureParams::tfRGBA:
        Format = DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
        break;

    };
    metadata.format = Format;
    metadata.depth = depth;
     
    DirectX::LoadFromDDSMemory(raw_data, w*h*4, DirectX::DDS_FLAGS_NONE, &metadata, image);
 
    DirectX::Convert(*image.GetImage(0, 1, 0), DXGI_FORMAT_BC7_UNORM, DirectX::TEX_FILTER_DEFAULT, 0, result);

   // DirectX::SaveToDDSFile(image.GetImage(0, 1, 0), DirectX::DDS_FLAGS::DDS_FLAGS_NONE, result.GetPixelsSize());
 
    const DirectX::Image * ptr = image.GetImage(0, 1, 0); // .pixels();

    if (ptr != nullptr)
    {
        u32 cnt = ptr->height* ptr->width;
        Msg("Textures[%d]", cnt);

        for (auto x = 0; x < ptr->width; x++)
        {
            for (auto y = 0; y < ptr->height; h++)
            {
                Msg_IN_FILE("row[%d][%d] = %d",  x, y, ptr->pixels[(x * y) + x]);
            }
        }
    }
    */

    // Bear Bundle
    /* CTimer T;
	T.Start();

	Msg("DXT: Compressing Image: %s %uX%u", out_name, w, h);

	R_ASSERT(0 != w && 0 != h);
	BearImage Image;
	BearTexturePixelFormat Format = BearTexturePixelFormat::R8G8B8A8;
	switch (fmt->fmt)
	{
		case STextureParams::tfDXT1: 	Format = BearTexturePixelFormat::BC1; 	  break;
		case STextureParams::tfADXT1:	Format = BearTexturePixelFormat::BC1a; 	  break;
		case STextureParams::tfDXT3: 	Format = BearTexturePixelFormat::BC2; 	  break;
		case STextureParams::tfDXT5: 	Format = BearTexturePixelFormat::BC3;	  break;
		case STextureParams::tfBC4: 	Format = BearTexturePixelFormat::BC4;	  break;
		case STextureParams::tfBC5: 	Format = BearTexturePixelFormat::BC5;	  break;
		case STextureParams::tfBC6: 	Format = BearTexturePixelFormat::BC6;	  break;
		case STextureParams::tfBC7: 	Format = BearTexturePixelFormat::BC7;	  break;
		case STextureParams::tfRGB: 	Format = BearTexturePixelFormat::R8G8B8;  break;
		case STextureParams::tfRGBA: 	Format = BearTexturePixelFormat::R8G8B8A8;break;
	}

	Image.Create(w, h, 1, 1, BearTexturePixelFormat::R8G8B8A8);
	bear_copy(*Image, raw_data, w * h * 4);
	Image.SwapRB();
	BearResizeFilter ResizeFilter = BearResizeFilter::Default;
	switch (fmt->mip_filter)
	{
	    case STextureParams::kMIPFilterBox:       ResizeFilter = BearResizeFilter::Box;     break;
	    case STextureParams::kMIPFilterTriangle:    ResizeFilter = BearResizeFilter::Triangle; break;
	    case STextureParams::kMIPFilterKaiser:     ResizeFilter = BearResizeFilter::Catmullrom;   break;
	}
 

    if (w <= 1024 && h <= 1024)
	{
		Image.GenerateMipmap(ResizeFilter);
		Image.Convert(Format);
	}
    

	Msg("DXT: Compressing Image: 2 [Closing File]. Time from start %f ms", T.GetElapsed_sec() * 1000.f);
	return Image.SaveToDds(out_name);;
 
	*/

    /// ORIGINAL 
 
   
     
    R_ASSERT(0 != w && 0 != h);
    bool result = false;
    nvtt::InputOptions inOpt;
    auto layout = fmt->type == STextureParams::ttCubeMap ? nvtt::TextureType_Cube : nvtt::TextureType_2D;
    inOpt.setTextureLayout(layout, w, h);
    inOpt.setMipmapGeneration(fmt->flags.is(STextureParams::flGenerateMipMaps));
    inOpt.setWrapMode(nvtt::WrapMode_Clamp);
    inOpt.setNormalMap(false);
    inOpt.setConvertToNormalMap(false);
    inOpt.setGamma(2.2f, 2.2f);
    inOpt.setNormalizeMipmaps(false);
    
    nvtt::CompressionOptions compOpt;
    compOpt.setQuality(nvtt::Quality_Highest);
    compOpt.setQuantization(fmt->flags.is(STextureParams::flDitherColor), false, fmt->flags.is(STextureParams::flBinaryAlpha));
    
    switch (fmt->fmt)
    {
        case STextureParams::tfDXT1:  compOpt.setFormat(nvtt::Format_DXT1 ); break;
        case STextureParams::tfADXT1: compOpt.setFormat(nvtt::Format_DXT1a); break;
        case STextureParams::tfDXT3:  compOpt.setFormat(nvtt::Format_DXT3 ); break;
        case STextureParams::tfDXT5:  compOpt.setFormat(nvtt::Format_DXT5 ); break;
        case STextureParams::tfBC7:   compOpt.setFormat(nvtt::Format_BC7); break;

        case STextureParams::tfRGB:   compOpt.setFormat(nvtt::Format_RGB  ); break;
        case STextureParams::tfRGBA:  compOpt.setFormat(nvtt::Format_RGBA ); break;
    }
    switch (fmt->mip_filter)
    {
        case STextureParams::kMIPFilterAdvanced: break;
        case STextureParams::kMIPFilterBox:      inOpt.setMipmapFilter(nvtt::MipmapFilter_Box     ); break;
        case STextureParams::kMIPFilterTriangle: inOpt.setMipmapFilter(nvtt::MipmapFilter_Triangle); break;
        case STextureParams::kMIPFilterKaiser:   inOpt.setMipmapFilter(nvtt::MipmapFilter_Kaiser  ); break;
    }


    nvtt::OutputOptions outOpt;
    outOpt.setFileName(out_name);

    DDSErrorHandler handler;
    outOpt.setErrorHandler(&handler);
      
    inOpt.setMipmapData(raw_data, w, h);

    // Msg("PTR input DDS: %p", raw_data);
    try 
    {
        result = nvtt::Compressor().process(inOpt, compOpt, outOpt);
    }
    catch (...)
    {
        Msg("Cant Convert DDS: %s", out_name);
        result = false;
    }
    
    if (!result)
    {
        //_unlink(out_name);
        return 0;
    }
 
    return 1;
}

#pragma optimize(on, "")

extern int DXTCompressBump(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);

extern "C" __declspec(dllexport) 
int   DXTCompress	(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth)
{
	switch (fmt->type){
	case STextureParams::ttImage:	
	case STextureParams::ttCubeMap: 
	case STextureParams::ttNormalMap:
	case STextureParams::ttTerrain:
		return DXTCompressImage	(out_name, raw_data, w, h, pitch, fmt, depth);
	break;
	case STextureParams::ttBumpMap: 
		return DXTCompressBump	(out_name, raw_data, normal_map, w, h, pitch, fmt, depth);
	break;
	default: NODEFAULT;
	}
	return -1;
}
