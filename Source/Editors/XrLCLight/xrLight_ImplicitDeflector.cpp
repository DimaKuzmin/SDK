#include "stdafx.h"
#include "xrlight_implicitdeflector.h"
#include "b_build_texture.h"
#include "xrFace.h"
#include "xrLC_GlobalData.h"

#include <xrDeflector.h>

extern "C" bool __declspec(dllimport)  
DXTCompress(LPCSTR out_name, u8* raw_data, u8* normal_map, u32 w, u32 h, u32 pitch, STextureParams* fmt, u32 depth);


u32	ImplicitDeflector::Width	()						
{
	return texture->dwWidth; 
}
u32	ImplicitDeflector::Height	()						
{
	return texture->dwHeight; 
}
	
u32&	ImplicitDeflector::Texel	(u32 x, u32 y)			
{
	u32* raw = static_cast<u32*>(*texture->pSurface);
	return raw[y*Width()+x];
}

void	ImplicitDeflector::Bounds	(u32 ID, Fbox2& dest)
{
	Face* F		= faces[ID];
	_TCF& TC	= F->tc[0];
	dest.min.set	(TC.uv[0]);
	dest.max.set	(TC.uv[0]);
	dest.modify		(TC.uv[1]);
	dest.modify		(TC.uv[2]);
}

void	ImplicitDeflector::Bounds_Summary (Fbox2& bounds)
{
	bounds.invalidate();
	for (u32 I=0; I<faces.size(); I++)
	{
		Fbox2	B;
		Bounds	(I,B);
		bounds.merge(B);
	}
}

void ImplicitDeflector::SaveTexture()
{
	// Expand
	CTimer tStats; tStats.Start();
	Status("Processing lightmap..."); 
	lmap.ApplyBordersFast(0);


	Msg("Apply Borders: %u ms", tStats.GetElapsed_ms());

	Status("Mixing lighting with texture...");
	{
		b_BuildTexture& TEX = *texture;
		if (TEX.pSurface.Empty())
			Msg("[ImplicitLighting] Problem Texture : %s, Detected", *TEX.name);
		u32* color = static_cast<u32*>(*TEX.pSurface);
		for (u32 V = 0; V < Height(); V++) {
			for (u32 U = 0; U < Width(); U++) {
				// Retreive Texel
				float	h = Lumel(U, V).h._r();
				u32& C = color[V * Width() + U];
				C = subst_alpha(C, u8_clr(h));
			}
		}
	}

	xr_vector<u32>				packed;
	lmap.Pack(packed);
	Deallocate();

	// base
	Status("Saving base...");
	{
		string_path				name, out_name;

		//se7kills rewrite		 
		sprintf(name, "%s", gCompilerMode.get_level_name());

		R_ASSERT(name[0] && texture);
		b_BuildTexture& TEX = *texture;
		strconcat(sizeof(out_name), out_name, name, "\\", TEX.name, ".dds");
		FS.update_path(out_name, "$game_levels$", out_name);
		clMsg("Saving texture '%s'...", out_name);
		VerifyPath(out_name);
		BYTE* raw_data = LPBYTE(*TEX.pSurface);
		u32	w = TEX.dwWidth;
		u32	h = TEX.dwHeight;
		u32	pitch = w * 4;
		STextureParams			fmt = TEX.THM;
		fmt.fmt = STextureParams::tfDXT5;
		fmt.flags.set(STextureParams::flDitherColor, FALSE);
		fmt.flags.set(STextureParams::flGenerateMipMaps, FALSE);
		fmt.flags.set(STextureParams::flBinaryAlpha, FALSE);
		DXTCompress(out_name, raw_data, 0, w, h, pitch, &fmt, 4);
	}

	// lmap
	Status("Saving lmap...");
	{
		string_path				name, out_name;
		sprintf(name, "%s", gCompilerMode.get_level_name());

		b_BuildTexture& TEX = *texture;
		strconcat(sizeof(out_name), out_name, name, "\\", TEX.name, "_lm.dds");
		FS.update_path(out_name, "$game_levels$", out_name);
		clMsg("Saving texture '%s'...", out_name);
		VerifyPath(out_name);
		BYTE* raw_data = LPBYTE(&*packed.begin());
		u32	w = TEX.dwWidth;
		u32	h = TEX.dwHeight;
		u32	pitch = w * 4;
		STextureParams			fmt;
		fmt.fmt = STextureParams::tfDXT5;
		fmt.flags.set(STextureParams::flDitherColor, FALSE);
		fmt.flags.set(STextureParams::flGenerateMipMaps, FALSE);
		fmt.flags.set(STextureParams::flBinaryAlpha, FALSE);
		DXTCompress(out_name, raw_data, 0, w, h, pitch, &fmt, 4);
	}
}

void ImplicitCalcGlobs::Initialize(ImplicitDeflector& def)
{
	defl = &def;
	Fbox2 bounds;
	defl->Bounds_Summary(bounds);
	Hash().initialize(bounds, defl->faces.size());
	for (u32 fid = 0; fid < defl->faces.size(); fid++)
	{
		Face* F = defl->faces[fid];

  		F->AddChannel(F->tc[0].uv[0], F->tc[0].uv[1], F->tc[0].uv[2]); // make compatible format with LMAPs
		defl->Bounds(fid, bounds);
		ImplicitHash->add(bounds, F);
	}
};