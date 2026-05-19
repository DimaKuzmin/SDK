#include "stdafx.h"

#include "lm_layer.h"
#include "ppl.h"
#include <atomic>


void lm_layer::Pack		(xr_vector<u32>& dest)const	
{
	dest.resize			(width*height);
	xr_vector<base_color>::const_iterator I=surface.begin();
	xr_vector<base_color>::const_iterator E=surface.end();
	xr_vector<u32>::iterator		W=dest.begin();
	for (; I!=E; I++)
	{
		base_color_c	C; I->_get(C);
		u8	_r	= u8_clr(C.rgb.x);
		u8	_g	= u8_clr(C.rgb.y);
		u8	_b	= u8_clr(C.rgb.z);
		u8	_d	= u8_clr(C.sun);
		*W++	= color_rgba(_r,_g,_b,_d);
	}
}
void lm_layer::Pack_hemi	(xr_vector<u32>& dest)const	//.
{
	dest.resize			(width*height);
	xr_vector<base_color>::const_iterator I=surface.begin	();
	xr_vector<base_color>::const_iterator E=surface.end	();
	xr_vector<u32>::iterator		W=dest.begin	();
	for (; I!=E; I++)
	{
		base_color_c	C;	I->_get(C);
		u8	_d	= u8_clr	(C.sun);
		u8	_h	= u8_clr	(C.hemi);
		//*W++	= color_rgba(_h,_h,_h,_d);
		*W++	= color_rgba(_d,_d,_d,_h);
	}
}
u32 lm_layer::Area()
{
	u8 BORDER = gCompilerMode.LC_lmap_BORDER;
	return (width + 2 * BORDER) * (height + 2 * BORDER);
}

void lm_layer::Pixel	(u32 ID, u8& r, u8& g, u8& b, u8& s, u8& h)
{
	xr_vector<base_color>::iterator I = surface.begin()+ID;
	base_color_c	c;	I->_get(c);
	r	= u8_clr(c.rgb.x);
	g	= u8_clr(c.rgb.y);
	b	= u8_clr(c.rgb.z);
	s	= u8_clr(c.sun);
	h	= u8_clr(c.hemi);
}

/*
	u32						width;
	u32						height;
	xr_vector<base_color>	surface;
	xr_vector<u8>			marker;
	LMODE					mode;	
*/

bool	lm_layer::similar			( const lm_layer &layer, float eps/* =EPS*/ ) const
{
	//if( mode != layer.mode )
		//return false;
	if( marker.size() != layer.marker.size() )
		return false;
	for( u32 i = 0; i< marker.size(); ++i )
	{
		if( marker[i]!=layer.marker[i] )
		{
			return false;
		}
	}
	if( surface.size() != layer.surface.size() )
		return false;
	for( u32 i = 0; i < surface.size(); ++i )
	{
		if( !surface[i].similar( layer.surface[i], EPS ) )
		{
			Msg("sufface diff id: %d", i);
			return false;
		}
	}

	return width ==  layer.width &&
		   height == layer.height;

}

bool lm_layer::ApplyBorders(u32 ref)
{
	bool bNeedContinue = false;

	// Копия surface для чтения (читаем старые значения)
	xr_vector<base_color> src = surface;
 
	// Сбор соседей
	base_color_c out;
	u32 idx, nidx;

	// Проходим по всем пикселям
	u32 count = 0;
	for (u32 y = 0; y < height; ++y)
	{
		for (u32 x = 0; x < width; ++x)
		{
			idx = y * width + x;
 
			// только пустые пиксели
			if (marker[idx] != 0) continue;

			count = 0; 
			out.clear_color();
			for (int dy = -1; dy <= 1; ++dy)
			{
				int ny = (int)y + dy;
				if (ny < 0 || ny >= (int)height) continue;

				for (int dx = -1; dx <= 1; ++dx)
				{
					int nx = (int)x + dx;
					if (nx < 0 || nx >= (int)width) continue;
					if (dx == 0 && dy == 0) continue; // сам пиксель

					nidx = ny * width + nx;
					if (marker[nidx] > ref)
					{
						base_color_c C;
						src[nidx]._get(C);
						out.add(C);
						count++;
					}
				}
			}

			if (count == 0)	continue;

			// усредняем
			out.scale(count);

			// геттер/сеттер base_color
			surface[idx]._set(out);
			marker[idx] = u8(std::min(ref, 255u));

			bNeedContinue = true;
		}
	} 
	return bNeedContinue;
}

struct Node
{
	u32 idx;
	u8 ref;
};

bool lm_layer::ApplyBordersFast(u32 checking_ref)
{
	// Замена 254м проходам 1м  !
	// for (auto ref = 254; ref > checking_ref; ref--)
	// 	if (!ApplyBorders(ref)) break;
	// return true;

	// Frontier метод
	const u32 W = width;
	const u32 H = height;

	xr_vector<Node> queue;
	queue.reserve(W * H / 4);

	for (u32 i = 0; i < W * H; ++i)
	{
		if (marker[i] != 0 && marker[i] > checking_ref)
			queue.push_back({ i, marker[i] });
	}

	size_t head = 0;
 	while (head < queue.size())
	{
		Node N = queue[head++];

		if (N.ref <= 1)
			continue;

		const s32 cx = N.idx % W;
		const s32 cy = N.idx / W;
 		const u8 nextRef = N.ref - 1;

		for (s32 dy = -1; dy <= 1; ++dy)
		{
			for (s32 dx = -1; dx <= 1; ++dx)
			{
 				if (dx == 0 && dy == 0)											
					continue;
 				const s32 nx = cx + dx;
				const s32 ny = cy + dy;
 				if (nx < 0 || ny < 0 || nx >= (s32)W || ny >= (s32)H)			
					continue;
 				const u32 nidx = ny * W + nx;
 				if (marker[nidx] != 0)											
					continue;
 
				// Среднее соседей !
				base_color_c accum; accum.clear_color();
 				u32 count = 0;
 				for (s32 sy = -1; sy <= 1; ++sy)
				{
					for (s32 sx = -1; sx <= 1; ++sx)
					{
						const s32 tx = nx + sx;
						const s32 ty = ny + sy;

						if (tx < 0 || ty < 0 || tx >= (s32)W || ty >= (s32)H)
							continue;

						const u32 tidx = ty * W + tx;
 						if (marker[tidx] > nextRef)
						{
							base_color_c C;
							surface[tidx]._get(C);
 							accum.add(C);
							++count;
						}
					}
				}

				if (count == 0)												
					continue;

				accum.scale(count);
 				surface[nidx]._set(accum);
				marker[nidx] = nextRef;

				if (nextRef > checking_ref)
					queue.push_back({ nidx, nextRef });
			}
		}
	}

	return true;
}