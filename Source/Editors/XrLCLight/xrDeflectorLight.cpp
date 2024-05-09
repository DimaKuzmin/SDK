#include "stdafx.h"

#include "xrdeflector.h"
//#include "build.h"
#include "cl_intersect.h"
#include "xrlc_globaldata.h"
//#include "std_classes.h"
#include "xrImage_Resampler.h"
#include "light_point.h"
#include "xrface.h"
#include "net_task.h"
#include "BuildArgs.h"
//const	u32	rms_discard			= 8;
//extern	BOOL		gl_linear	;

void Jitter_Select(Fvector2* &Jitter, u32& Jcount)
{
	static Fvector2 Jitter1[1] = {
		{0,0}
	};
	static Fvector2 Jitter4[4] = {
		{-1,-1}, {1,-1}, {1,1}, {-1,1}
	};
	static Fvector2 Jitter9[9] = {
		{-1,-1},	{0,-1},		{1,-1}, 
		{-1,0},		{0,0},		{1,0},
		{-1,1},		{0,1},		{1,1}
	};

	switch (g_params().m_lm_jitter_samples)
	{
	case 1:
		Jcount	= 1;
		Jitter	= Jitter1;
		break;
	case 9:
		Jcount	= 9;
		Jitter	= Jitter9;
		break;
	case 4:
	default:
		Jcount	= 4;
		Jitter	= Jitter4;
		break;
	}
}


void GET(const base_color &surface_color,
		 const u8 marker,
		//u32 width, u32 height, int x,  int y, 
		u32 ref, u32 &count,  base_color_c& dst )
{
	//if (x<0) return;
	//else if (x>=(int)width)		return;
	//if (y<0) return;
	//else if (y>=(int)height)	return;

	// summarize
	//u32		id	= y*width + x;
	if (marker<=ref)		return;

	base_color_c		C;
	surface_color._get	(C);
	dst.add				(C);
	count				++;
}

void GET			(const lm_layer &lm, int x, int y, u32 ref, u32 &count,  base_color_c& dst)
{
	// wrap pixels
	if (x<0) return;
	else if (x>=(int)lm.width)	return;
	if (y<0) return;
	else if (y>=(int)lm.height)	return;

	// summarize
	u32		id	= y*lm.width + x;
	if (lm.marker[id]<=ref)		return;

	base_color_c		C;
	lm.surface[id]._get	(C);
	dst.add				(C);
	count				++;
}
 
/*
struct	Get8_res
{
	Get8_res():count(0), res_surface_color(0), res_marker(0), ref( u32(-1) ) {}
	base_color_c	dst;
	u32				count;
	base_color		*res_surface_color;
	u8				*res_marker;
	u32				ref;
	void aplly( int x, int y, u32 _ref, bool &bNeedContinue, lm_layer &result )
	{
		if (count) {
			dst.scale			(count);
			result.surface		[y*result.width+x]._set(dst);
			result.marker		[y*result.width+x]=u8(_ref);
			bNeedContinue		= TRUE;
		}
	}
	void apply_delay(int x, int y, u32 _ref, bool &bNeedContinue, lm_layer &result )
	{
		if (count) {
			dst.scale			(count);
			res_surface_color = &(result.surface		[y*result.width+x]);//._set(dst);
			res_marker		  = &(result.marker		[y*result.width+x]);//=u8(ref);
			ref				  = _ref;
			bNeedContinue		= TRUE;
		}
	}
	~Get8_res()
	{
		if( res_surface_color )
		{
			R_ASSERT( res_marker );
			R_ASSERT( ref != u32(-1) );
			res_surface_color->_set(dst);
			*res_marker		 =u8(ref);
		}
	}
};

void Get8(const lm_layer &lm, int x, int y, u32 ref, u32 &count,  base_color_c& dst )
{
		GET(lm,x-1,y-1,ref,count,dst);
		GET(lm,x  ,y-1,ref,count,dst);
		GET(lm,x+1,y-1,ref,count,dst);

		GET(lm,x-1,y  ,ref,count,dst);
		GET(lm,x+1,y  ,ref,count,dst);

		GET(lm,x-1,y+1,ref,count,dst);
		GET(lm,x  ,y+1,ref,count,dst);
		GET(lm,x+1,y+1,ref,count,dst);
}

void Get8(lm_layer &lm, int x, int y, u32 ref, Get8_res &res )
{
	Get8( lm, x, y, ref, res.count, res.dst );
}
*/
 
struct lm_line
{
 buffer_vector<base_color>	&surface;
 buffer_vector<u8>			&marker;
 u32 y;
 u32 height;
 lm_line( buffer_vector<base_color>& surf_buf, buffer_vector<u8>& mark_buf ):
	surface( surf_buf ), marker( mark_buf ), y(u32(-1)), height(u32(-1))
 {}
 void save( int _y, const lm_layer &lm )
	{
		y		= _y;
		height	= lm.height;

		{
			xr_vector<base_color>::const_iterator from = lm.surface.begin() + y*lm.width;
			xr_vector<base_color>::const_iterator to	=  from + lm.width;
			surface.assign( from, to );
		}

		{
			xr_vector<u8>::const_iterator from = lm.marker.begin() + y*lm.width;
			xr_vector<u8>::const_iterator to	=  from + lm.width;
			marker.assign( from, to );
		}

		//surface.resize( lm.width );
		//marker.resize(  lm.width );
		//for (int x=0; x<(int)lm.width; x++)
		//{
		//	surface[x] = lm.surface[y*lm.width+x];
		//	marker[x]	= lm.marker[y*lm.width+x];
		//}
	}
} ;
void GET		( const lm_line& l, int x, u32 width, u32 ref,  u32 &count,  base_color_c& dst  )
{
	if (x<0) return;
	else if (x>=(int)width)			return;
	if (l.y<0) return;
	else if (l.y>=(int)l.height)	return;

	// summarize
	u32		id	=  x;
	if (l.marker[id]<=ref)		return;

	base_color_c		C;
	l.surface[id]._get	(C);
	dst.add				(C);
	count				++;
}



BOOL NEW_ApplyBorders	(lm_layer &lm, u32 ref) 
{
	bool			bNeedContinue = false;
	
	buffer_vector<base_color>	buf_surf_line0( _alloca( lm.width*sizeof( base_color ) ), lm.width );
	
	buffer_vector<base_color>	buf_surf_line1( _alloca( lm.width*sizeof( base_color ) ), lm.width );

	buffer_vector<u8>			buf_marker_line0( _alloca( lm.width*sizeof( u8 ) ), lm.width );

	buffer_vector<u8>			buf_marker_line1( _alloca( lm.width*sizeof( u8 ) ), lm.width );


	lm_line line0( buf_surf_line0, buf_marker_line0 );
	lm_line line1( buf_surf_line1, buf_marker_line1 );

	try {
		//lm_layer	result	= lm;

		lm_line *l_0  = &line0;
		lm_line *l_1  = &line1;

		for (int y=0; y<(int)lm.height; y++) {
	
			l_0->save( y, lm );

			std::swap( l_0, l_1 );

			lm_line &line = *l_0;

			base_color sv_color0; sv_color0._set( -1,-1,-1 );
			u8		   sv_marker0 = u8(-1);
			for (int x=0; x<(int)lm.width; x++)
			{
				base_color sv_color  = sv_color0;
				u8		   sv_marker = sv_marker0;
				sv_color0  = lm.surface[y*lm.width+x];
				sv_marker0 = lm.marker[y*lm.width+x];
				if (lm.marker[y*lm.width+x]==0) 
				{

					base_color_c	clr;
					u32			C	=0;
					if( y > 0 )
					{
						GET(line,x-1,lm.width,ref,C,clr);
						GET(line,x  ,lm.width,ref,C,clr);
						GET(line,x+1,lm.width,ref,C,clr);
					}
					//GET(lm,x-1,y-1,ref,C,clr);
					//GET(lm,x  ,y-1,ref,C,clr);
					//GET(lm,x+1,y-1,ref,C,clr);

					//GET(lm,x-1,y  ,ref,C,clr);
					if( x > 0 )
						GET( sv_color, sv_marker, ref, C, clr );



					GET(lm,x+1,y  ,ref,C,clr);

					GET(lm,x-1,y+1,ref,C,clr);
					GET(lm,x  ,y+1,ref,C,clr);
					GET(lm,x+1,y+1,ref,C,clr);
					
					if (C)
					{
						clr.scale			(C);
						lm.surface		[y*lm.width+x]._set(clr);
						lm.marker		[y*lm.width+x]=u8(ref);
						bNeedContinue		= TRUE;
					}
					
				}
			}
		}
		//lm	= result;
	} catch (...)
	{
		clMsg("* ERROR: ApplyBorders");
	}
	return bNeedContinue;
}



BOOL OLD_ApplyBorders	(lm_layer &lm, u32 ref) 
{
	BOOL			bNeedContinue = FALSE;

	try {
		lm_layer	result	= lm;

		for (int y=0; y<(int)lm.height; y++) {
			for (int x=0; x<(int)lm.width; x++)
			{
				if (lm.marker[y*lm.width+x]==0) 
				{
					base_color_c	clr;
					u32			C	=0;
					GET(lm,x-1,y-1,ref,C,clr);
					GET(lm,x  ,y-1,ref,C,clr);
					GET(lm,x+1,y-1,ref,C,clr);

					GET(lm,x-1,y  ,ref,C,clr);
					GET(lm,x+1,y  ,ref,C,clr);

					GET(lm,x-1,y+1,ref,C,clr);
					GET(lm,x  ,y+1,ref,C,clr);
					GET(lm,x+1,y+1,ref,C,clr);

					if (C) {
						clr.scale			(C);
						result.surface		[y*lm.width+x]._set(clr);
						result.marker		[y*lm.width+x]=u8(ref);
						bNeedContinue		= TRUE;
					}
				}
			}
		}
		lm	= result;
	} catch (...)
	{
		clMsg("* ERROR: ApplyBorders");
	}
	return bNeedContinue;
}

BOOL ApplyBorders( lm_layer &lm, u32 ref ) 
{
	
	//lm_layer r_new = lm;
	//BOOL bres_new = NEW_ApplyBorders( r_new, ref );
	//lm_layer r_old = lm;
	//BOOL bres_old = OLD_ApplyBorders( r_old, ref );

	//R_ASSERT( r_old.similar( r_new, 0 ) );
	//R_ASSERT( bres_new == bres_old );

	//
	//lm = r_new;
	//return bres_new;
	//
	
	return NEW_ApplyBorders( lm, ref );
}
 
// OLDER GARBAGE

extern u64 RayID = 0;
  
 
bool readed = false;
bool return_parrams = false;

 

float getLastRP_Scale(CDB::COLLIDER* DB, CDB::MODEL* MDL, R_Light& L, Face* skip, BOOL bUseFaceDisable)
{

	u32		tris_count	= DB->r_count();
	float	scale		= 1.f;
	Fvector B;

	X_TRY 
	{
		for (u32 I=0; I<tris_count; I++)
		{
			CDB::RESULT& rpinf = DB->r_begin()[I];
 
			// Access to texture
			CDB::TRI& clT										= MDL->get_tris()[rpinf.id];
			base_Face* F										= (base_Face*)(clT.pointer);
			if (0==F)											continue;
			if (skip==F)										continue;

			const Shader_xrLC&	SH								= F->Shader();
			if (!SH.flags.bLIGHT_CastShadow || F->flags.bShadowSkip)					continue; //
 
			if (F->flags.bOpaque)	
			{
				// Opaque poly - cache it
				L.tri[0].set	(rpinf.verts[0]);
				L.tri[1].set	(rpinf.verts[1]);
				L.tri[2].set	(rpinf.verts[2]);
				return 0;
			}

			b_material& M	= inlc_global_data()->materials()			[F->dwMaterial];
			b_texture&	T	= inlc_global_data()->textures()			[M.surfidx];

#ifdef		DEBUG
			const b_BuildTexture	&build_texture  = inlc_global_data()->textures()			[M.surfidx];

			VERIFY( !!(build_texture.THM.HasSurface()) ==  !!(!T.pSurface.Empty()) );
#endif

			if (T.pSurface.Empty())	
			{
				F->flags.bOpaque	= true;
			
				clMsg			("* ERROR: RAY-TRACE: Strange face detected... Has alpha without texture...: %s", build_texture.name);
				return 0;
			}

			// barycentric coords
			// note: W,U,V order
			B.set	(1.0f - rpinf.u - rpinf.v, rpinf.u, rpinf.v);

			// calc UV
			Fvector2*	cuv = F->getTC0					();
			Fvector2	uv;
			uv.x = cuv[0].x*B.x + cuv[1].x*B.y + cuv[2].x*B.z;
			uv.y = cuv[0].y*B.x + cuv[1].y*B.y + cuv[2].y*B.z;

			int U = iFloor(uv.x*float(T.dwWidth) + .5f);
			int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
			U %= T.dwWidth;		if (U<0) U+=T.dwWidth;
			V %= T.dwHeight;	if (V<0) V+=T.dwHeight;
			u32* raw = static_cast<u32*>(*T.pSurface);
			u32 pixel		= raw[V*T.dwWidth+U];
			u32 pixel_a		= color_get_A(pixel);
			float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
			scale			*= opac;
		}
	} 
	X_CATCH
	{
		clMsg("* ERROR: getLastRP_Scale");
	}

	return scale;
}

// NEW CDB_RAY
void FilterFunction(OpcodeArgs* args)
{
	CDB::MODEL* MDL = (CDB::MODEL* ) args->MDL;

	// Access to texture
	CDB::TRI& clT										= MDL->get_tris()[args->hit_struct.prim];
	base_Face* F										= (base_Face*)(clT.pointer);
	
	//if (RayID < 100)
	//	Msg("[OPCODE] Ray[%d] TFar[%f] prim[%llu] energy[%f]", RayID, args->hit_struct.dist, args->hit_struct.prim, args->energy);
 
	if (0==F || args->skip==F)							
		return;
 
	const Shader_xrLC&	SH							= F->Shader();
	if (!SH.flags.bLIGHT_CastShadow || F->flags.bShadowSkip) //
		return;
 
	if (F->flags.bOpaque)	
	{
		R_Light& light = (* ( (R_Light*) args->Light) );
 
		// Opaque poly - cache it
		light.tri[0].set	(MDL->get_verts()[clT.verts[0]]);
		light.tri[1].set	(MDL->get_verts()[clT.verts[1]]);
		light.tri[2].set	(MDL->get_verts()[clT.verts[2]]);
		
		args->valid = false;
		args->energy = 0;
 		return;
	}

	b_material& M	= inlc_global_data()->materials()			[F->dwMaterial];
	b_texture&	T	= inlc_global_data()->textures()			[M.surfidx];
 
	if (T.pSurface.Empty())	
	{
		F->flags.bOpaque	= true;

		clMsg			("* ERROR: RAY-TRACE: Strange face detected... Has alpha without texture... %s", T.name);
		
		args->valid = false;
		args->energy = 0;
		return;
	}

	//args->hits.push_back(args->hit_struct);
 
	// barycentric coords
	// note: W,U,V order
	Fvector B;
	B.set	(1.0f - args->hit_struct.u - args->hit_struct.v, args->hit_struct.u, args->hit_struct.v);

	// calc UV
	Fvector2*	cuv = F->getTC0					();
	Fvector2	uv;
	uv.x = cuv[0].x*B.x + cuv[1].x*B.y + cuv[2].x*B.z;
	uv.y = cuv[0].y*B.x + cuv[1].y*B.y + cuv[2].y*B.z;

	int U = iFloor(uv.x*float(T.dwWidth) + .5f);
	int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
	U %= T.dwWidth;	
	if (U<0) U+=T.dwWidth;
	V %= T.dwHeight;
	if (V<0) V+=T.dwHeight;

	u32* raw = static_cast<u32*>(*T.pSurface);
	u32 pixel		= raw[V*T.dwWidth+U];
	u32 pixel_a		= color_get_A(pixel);
	float opac		= 1.f - _sqr(float(pixel_a)/255.f);
 
	args->energy *= opac;

	// Energy Dead
	if (args->energy < 0.001f)
		args->valid = false; 


};
  
float rayTraceCheck	(CDB::COLLIDER* DB, CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& D, float R, Face* skip)
{ 
	R_ASSERT	(DB);
 
	// 1. Check cached polygon	 
	float _u,_v,range;
	bool res = CDB::TestRayTri(P,D,L.tri,_u,_v,range,false);
	if (res) 
	if (range > 0 && range < R) 
	{
 		return 0;
	}
 
  	// 2. Polygon doesn't pick - real database query
     
	OpcodeContext ctxt;
	ctxt.filter = &FilterFunction;
	ctxt.r_dir = D;
	ctxt.r_start = P;
	ctxt.r_range = R;
	

	OpcodeArgs args;
	args.energy = 1.0f;
 	args.Light	= (void*) &L;
	args.skip	= (void*) skip;
   	args.MDL	= (void*) MDL;
	args.valid  = 1;

	ctxt.result = &args;
 
	DB->rayTrace1(&ctxt);  	 

	return ctxt.result->energy;

	/*
	if (ctxt.result->energy == 0)
		return 0;
	
	float result = 1.0f;
 
	Fvector B;
	for (auto& hit : args.hits)
	{
		CDB::TRI& clT										= MDL->get_tris()[hit.prim];
		base_Face* F										= (base_Face*)(clT.pointer);
 		
		b_material& M	= inlc_global_data()->materials()			[F->dwMaterial];
		b_texture&	T	= inlc_global_data()->textures()			[M.surfidx];
		
		B.set(1.0f - hit.u - hit.v, hit.u, hit.v);

		// calc UV
		Fvector2*	cuv = F->getTC0					();
		Fvector2	uv;
		uv.x = cuv[0].x*B.x + cuv[1].x*B.y + cuv[2].x*B.z;
		uv.y = cuv[0].y*B.x + cuv[1].y*B.y + cuv[2].y*B.z;

		int U = iFloor(uv.x*float(T.dwWidth) + .5f);
		int V = iFloor(uv.y*float(T.dwHeight)+ .5f);
		U %= T.dwWidth;	
		if (U<0) U+=T.dwWidth;
		V %= T.dwHeight;
		if (V<0) V+=T.dwHeight;

		u32* raw = static_cast<u32*>(*T.pSurface);
		u32 pixel		= raw[V*T.dwWidth+U];
		u32 pixel_a		= color_get_A(pixel);
		float opac		= 1.f - _sqr(float(pixel_a)/255.f);
		result *= opac;
 	}
	

	return result;
	*/
}
 
float RaytraceEmbreeProcess(CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& N, float range, Face* skip);

extern XRLC_LIGHT_API SpecialArgsXRLCLight* build_args;
 
float rayTrace	(CDB::COLLIDER* DB, CDB::MODEL* MDL, R_Light& L, Fvector& P, Fvector& D, float R, Face* skip, BOOL bUseFaceDisable, bool use_opcode)
{ 	
	RayID++;

	if (build_args->use_embree && !use_opcode)
		return RaytraceEmbreeProcess(MDL, L, P, D, R, skip);
 	
	if (!build_args->use_opcode_old)
		return rayTraceCheck(DB, MDL, L, P, D, R, skip);


	R_ASSERT	(DB);
 
	// 1. Check cached polygon
	 
	float _u,_v,range;
	bool res = CDB::TestRayTri(P,D,L.tri,_u,_v,range,false);
	if (res) 
	if (range > 0 && range < R) 
		return 0;
  
	Fvector pos = P;
	Fvector dir = D;
	float Range = R;

  	// 2. Polygon doesn't pick - real database query
	//Msg("Orig Pos{%f,%f,%f}, dir{%f,%f,%f}, range[%f]", VPUSH(pos), VPUSH(dir), Range);
	DB->ray_query(MDL, P, D, R);

	// 3. Analyze polygons and cache nearest if possible
	if (0==DB->r_count()) 
		return 1;
	else
	{
		//Msg("Embree Pos{%f,%f,%f}, dir{%f,%f,%f}, range[%f]", VPUSH(pos), VPUSH(dir), Range);
		

		return getLastRP_Scale(DB, MDL, L, skip, bUseFaceDisable);
	}

	return 0;
 
}

#define USE_RGB_OPCODE false
#define USE_SUN_OPCODE false
#define USE_HEMI_OPCODE false

#include "EmbreeDataStorage.h"

void RayTraceEmbree8Preocess(PackedBuffer* buffer, ELightType type_lightpoint, ELights type_LIGHTs);
 
void LightPointPacked(PackedBufferTOProcess* buffer_to_work, base_lighting& lights, u32 flags)
{
	Fvector		Ldir;
 
	/*
	if (0 == (buffer->flags[0] & LP_dont_rgb))
	{
		R_Light* L = &*lights.rgb.begin(), * E = &*lights.rgb.end();
		for (; L != E; L++)
		{
			buffer->light = L;

			switch (L->type)
			{
				case LT_DIRECT:
				{
					int valid[8];
					bool any = false;
					
					for (auto i = 0; i < 8; i++)
					{
						Pnew.mad(buffer->position[i], buffer->direction[i], 0.01f);

						// Cos
						Ldir.invert(L->direction);
						float D = Ldir.dotproduct(buffer->direction[i]);
						
						valid[i] = false;
						buffer->Dist2Light[i] = D;

						if (D <= 0) 
							continue;
						
						valid[i] = true;
						any = true;
					}

					if (any)
					{
 						// Trace Light
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}
				}
				break;
				case LT_POINT:
				{
					int valid[8];
					bool any = false;
					 
					for (auto i = 0; i < 8; i++)
					{
						Pnew.mad(buffer->position[i], buffer->direction[i], 0.01f);

						// Distance
						float sqD = buffer->position[i].distance_to_sqr(L->position);
						valid[i] = false;
						if (sqD > L->range2)
							continue;
						
						// Dir
						Ldir.sub(L->position, buffer->position[i]);
						Ldir.normalize_safe();
						float D = Ldir.dotproduct(buffer->direction[i]);
						if (D <= 0)
							continue;
						
						valid[i] = true;
						any = true;
						buffer->Dist2Light[i] = D;

					}

					if (any)
					{
						// Trace Light
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}
 
				}
				break;
				case LT_SECONDARY:
				{
					int valid[8];
					bool any = false;

					for (auto i = 0; i < 8; i++)
					{
						// Distance
						valid[i] = false;
 
						float sqD = buffer->position[i].distance_to_sqr(L->position);
						if (sqD > L->range2) continue;

						// Dir
						Ldir.sub(L->position, buffer->position[i]);
						Ldir.normalize_safe();
						float	D = Ldir.dotproduct(buffer->direction[i]);
						if (D <= 0) continue;

						D *= -Ldir.dotproduct(L->direction);
						if (D <= 0) continue;
						
						valid[i] = false;
						buffer->Dist2Light[i] = D;
						any = true;
					}

					if (any)
					{
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}
				}
				break;
			}
		}
	}
	*/
 

	 
	R_Light* L = &*lights.hemi.begin(), * E = &*lights.hemi.end();
	
	for (; L != E; L++)
	{
		PackedBuffer buffer;
 		buffer.MDL = buffer_to_work->MDL;
		buffer.light = L;
		buffer.flags = flags;
		
		
		
 		if (L->type == LT_DIRECT)
		{
 			bool any = false;

			for (int i = 0; i < 8; i++)
			{
 				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(buffer_to_work->direction[i]);
				buffer.valid[i] = -1;

				if (D <= 0) 
					continue;

				any = true;
				buffer.valid[i] = 0;
				
				// Trace Light
				Fvector	PMoved, PNew;
				PNew.set(buffer_to_work->position[i]); PNew.add(0.01f);
				PMoved.mad(PNew, Ldir, 0.001f);
				
				buffer.pos[i].set(PMoved);
				buffer.dir[i].set(Ldir);
				buffer.tmax[i] = 1000.f;
				buffer.skip[i] = buffer_to_work->skip[i];
			}

			if (any)
			{
				RayTraceEmbree8Preocess(&buffer, ELightType::LT_Direct, ELights::Hemi);

				for (auto i = 0; i < 8; i++)
				{
					buffer_to_work->color[i].add(buffer.Color[i]);
				}
			}
			
		}
		else
		{
 			bool any = false;

			for (int i = 0; i < 8; i++)
			{
				buffer.valid[i] = -1;
				// Distance
				float sqD = buffer_to_work->position[i].distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, buffer_to_work->position[i]);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(buffer_to_work->direction[i]);
				if (D <= 0) continue;


				any = true;
				buffer.valid[i] = 0;

				// Trace Light
				float R = _sqrt(sqD);
				
				Fvector	PNew;
				PNew.set(buffer_to_work->position[i]); PNew.add(0.01f);
   
				buffer.pos[i].set(PNew);
				buffer.dir[i].set(Ldir);
				buffer.tmax[i] = R;
				buffer.skip[i] = buffer_to_work->skip[i];

			}

			if (any)
			{
				RayTraceEmbree8Preocess(&buffer, ELightType::LT_Point, ELights::Hemi);

				for (auto i = 0; i < 8; i++)
				{
					buffer_to_work->color[i].add(buffer.Color[i]);
				}
			}
 
		}

	}
	 
}


IC void LightPoint(CDB::COLLIDER* DB, CDB::MODEL* MDL, base_color_c &C, Fvector &P, Fvector &N, base_lighting& lights, u32 flags, Face* skip, bool use_opcode)
{
	Fvector		Ldir,Pnew;
	Pnew.mad	(P,N,0.01f);

	BOOL		bUseFaceDisable	= flags&LP_UseFaceDisable;
 
	if (0==(flags&LP_dont_rgb))
	{
		 
		DB->ray_options	(0);
		R_Light	*L	= &*lights.rgb.begin(), *E = &*lights.rgb.end();
		for (;L!=E; L++)
		{
			switch (L->type)
			{
				case LT_DIRECT:
				{
					// Cos
					Ldir.invert	(L->direction);
					float D		= Ldir.dotproduct( N );
					if( D <=0 ) continue;

					// Trace Light
					float scale	=	D * L->energy * rayTrace(DB, MDL, *L, Pnew, Ldir, 1000.f, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode);  
					C.rgb.x		+=	scale * L->diffuse.x; 
					C.rgb.y		+=	scale * L->diffuse.y;
					C.rgb.z		+=	scale * L->diffuse.z;
				}
				break;
				case LT_POINT:
				{
					// Distance
					float sqD	=	P.distance_to_sqr	(L->position);
					if (sqD > L->range2) continue;

					// Dir
					Ldir.sub			(L->position,P);
					Ldir.normalize_safe	();
					float D				= Ldir.dotproduct( N );
					if( D <=0 )			continue;

					// Trace Light
					float R		= _sqrt(sqD);
					float scale = D*L->energy * rayTrace(DB,MDL, *L, Pnew, Ldir, R, skip, bUseFaceDisable, USE_RGB_OPCODE || use_opcode); 
					float A		;
					
					if (inlc_global_data()->gl_linear())
					{
						A = 1 - R / L->range;
					}
					else
					{
						//	Igor: let A equal 0 at the light boundary
						A	= scale * 
							(
								1/(L->attenuation0 + L->attenuation1*R + L->attenuation2*sqD) - 
								R*L->falloff
							);

					}

					C.rgb.x += A * L->diffuse.x;
					C.rgb.y += A * L->diffuse.y;
					C.rgb.z += A * L->diffuse.z;
				}
				break;
				case LT_SECONDARY:
				{
					// Distance
					float sqD	=	P.distance_to_sqr	(L->position);
					if (sqD > L->range2) continue;

					// Dir
					Ldir.sub	(L->position,P);
					Ldir.normalize_safe();
					float	D	=	Ldir.dotproduct		( N );
					if( D <=0 ) continue;
					D	*=	-Ldir.dotproduct	(L->direction);
					if( D <=0 ) continue;

					// Jitter + trace light -> monte-carlo method
					Fvector	Psave	= L->position, Pdir;
					L->position.mad	(Pdir.random_dir(L->direction,PI_DIV_4),.05f);
					
					float R			= _sqrt(sqD);
					float scale		= powf(D, 1.f/8.f)*L->energy * rayTrace(DB,MDL, *L,Pnew,Ldir,R,skip,bUseFaceDisable, USE_RGB_OPCODE || use_opcode);	  
					float A			= scale * (1-R/L->range);
					L->position		= Psave;

					C.rgb.x += A * L->diffuse.x;
					C.rgb.y += A * L->diffuse.y;
					C.rgb.z += A * L->diffuse.z;
				}
				break;
			}
		}
	}

	if (0==(flags&LP_dont_sun))
	{
		DB->ray_options	(0);
		R_Light	*L		= &*(lights.sun.begin()), *E = &*(lights.sun.end());
		for (;L!=E; L++)
		{
			if (L->type==LT_DIRECT)
			{
				// Cos
				Ldir.invert	(L->direction);
				float D		= Ldir.dotproduct( N );
				if( D <=0 ) continue;

				// Trace Light
				float scale	=	L->energy*rayTrace(DB,MDL, *L,Pnew,Ldir,1000.f,skip,bUseFaceDisable, USE_SUN_OPCODE || use_opcode);
				C.sun		+=	scale;
			} 
			else 
			{
				// Distance
				float sqD	=	P.distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub			(L->position,P);
				Ldir.normalize_safe	();
				float D				= Ldir.dotproduct( N );
				if( D <=0 )			continue;

				// Trace Light
				float R		=	_sqrt(sqD);
				float scale =	D*L->energy*rayTrace(DB,MDL, *L,Pnew,Ldir,R,skip,bUseFaceDisable,  USE_SUN_OPCODE || use_opcode);
				float A		=	scale / (L->attenuation0 + L->attenuation1*R + L->attenuation2*sqD);

				C.sun		+=	A;
			}
		}
	}

	if (0==(flags&LP_dont_hemi))
	{
		R_Light	*L	= &*lights.hemi.begin(), *E = &*lights.hemi.end();
		for (;L!=E; L++)
		{
			if (L->type==LT_DIRECT) 
			{	
 				// Cos
				Ldir.invert	(L->direction);
				float D		= Ldir.dotproduct( N );
				if( D <=0 ) continue;

				// Trace Light
				Fvector		PMoved;
				PMoved.mad	(Pnew,Ldir,0.001f);
				float scale	=	L->energy*rayTrace(DB,MDL, *L,PMoved,Ldir,1000.f,skip,bUseFaceDisable, USE_HEMI_OPCODE || use_opcode);
				C.hemi		+=	scale;
 			}
			else
			{
 				// Distance
				float sqD	=	P.distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub			(L->position,P);
				Ldir.normalize_safe	();
				float D		=	Ldir.dotproduct( N );
				if( D <=0 ) continue;

				// Trace Light
				float R		=	_sqrt(sqD);
				float scale =	D*L->energy*rayTrace(DB,MDL, *L,Pnew,Ldir,R,skip,bUseFaceDisable, USE_HEMI_OPCODE ||  use_opcode);
				float A		=	scale / (L->attenuation0 + L->attenuation1*R + L->attenuation2*sqD);

				C.hemi		+=	A;
			}
 
		}
	}
}

IC u32	rms_diff	(u32 a, u32 b)
{
	if (a>b)	
		return a-b;
	else
		return b-a;
}

BOOL	__stdcall rms_test	(lm_layer& lm, u32 w, u32 h, u32 rms)
{
	if ((w<=1) || (h<=1))	return FALSE;

	// scale down(lanczos3) and up (bilinear, as video board) //.
	xr_vector<u32>	pOriginal_base;	lm.Pack					(pOriginal_base);
	xr_vector<u32>	pScaled_base;	pScaled_base.resize		(w*h);
	xr_vector<u32>	pRestored_base;	pRestored_base.resize	(lm.width*lm.height);
	xr_vector<u32>	pOriginal_hemi;	lm.Pack_hemi			(pOriginal_hemi);
	xr_vector<u32>	pScaled_hemi;	pScaled_hemi.resize		(w*h);
	xr_vector<u32>	pRestored_hemi;	pRestored_hemi.resize	(lm.width*lm.height);

	try
	{
		// rgb + sun
		imf_Process	(&*pScaled_base.begin(),	w,			h,			&*pOriginal_base.begin(),	lm.width,lm.height,imf_lanczos3	);
		imf_Process	(&*pRestored_base.begin(),	lm.width,	lm.height,	&*pScaled_base.begin(),		w,h,imf_filter					);
		// hemi
		//.
		/*
		if ((lm.width/2>1)&&(lm.height/2>1)){
			imf_Process	(&*pRestored_hemi.begin(),	lm.width/2,	lm.height/2,&*pOriginal_hemi.begin(),	lm.width,lm.height,		imf_lanczos3	);
			imf_Process	(&*pOriginal_hemi.begin(),	lm.width,	lm.height,	&*pRestored_hemi.begin(),	lm.width/2,	lm.height/2,imf_filter		);
		}
		*/
		imf_Process	(&*pScaled_hemi.begin(),	w,			h,			&*pOriginal_hemi.begin(),	lm.width,lm.height,imf_lanczos3	);
		imf_Process	(&*pRestored_hemi.begin(),	lm.width,	lm.height,	&*pScaled_hemi.begin(),		w,h,imf_filter					);
	}
	catch (...)
	{
		clMsg	("* ERROR: imf_Process");
		return	FALSE;
	}

	// compare them
	const u32 limit = 254-BORDER;
	for (u32 y=0; y<lm.height; y++)
	{
		u32		offset			= y*lm.width;
		u8*		scan_mark		= (u8*)	&*(lm.marker.begin()+offset);		//.
		u32*	scan_lmap_base	= (u32*)&*(pOriginal_base.begin()+offset);	
		u32*	scan_rest_base	= (u32*)&*(pRestored_base.begin()+offset);	
		u32*	scan_lmap_hemi	= (u32*)&*(pOriginal_hemi.begin()+offset);	
		u32*	scan_rest_hemi	= (u32*)&*(pRestored_hemi.begin()+offset);	
		for (u32 x=0; x<lm.width; x++){
			if (scan_mark[x]>=limit){
				u32 pixel_base		= scan_lmap_base[x];
				u32 pixel_r_base	= scan_rest_base[x];
				u32 pixel_hemi		= scan_lmap_hemi[x];
				u32 pixel_r_hemi	= scan_rest_hemi[x];
				if (rms_diff(color_get_R(pixel_r_base),color_get_R(pixel_base))>rms)			return FALSE;
				if (rms_diff(color_get_G(pixel_r_base),color_get_G(pixel_base))>rms)			return FALSE;
				if (rms_diff(color_get_B(pixel_r_base),color_get_B(pixel_base))>rms)			return FALSE;
				if (rms_diff(color_get_A(pixel_r_base),color_get_A(pixel_base))>rms)			return FALSE;
				if (rms_diff(color_get_R(pixel_r_hemi),color_get_R(pixel_hemi))>((rms*4)/3))	return FALSE;
			}
		}
	}
	return	TRUE;
}

BOOL	__stdcall rms_test	(lm_layer&	lm, u32 _r, u32 _g, u32 _b, u32 _s, u32 _h, u32 rms)
{
	u32 x,y;
	for (y=0; y<lm.height; y++)
	{
		for (x=0; x<lm.width; x++)
		{
			u32	offset		= y*lm.width+x;
			if (lm.marker[offset]>=254)	{
				u8			r,g,b,s,h;
				lm.Pixel	(offset,r,g,b,s,h);
				if (rms_diff(_r, r)>rms)			return FALSE;
				if (rms_diff(_g, g)>rms)			return FALSE;
				if (rms_diff(_b, b)>rms)			return FALSE;
				if (rms_diff(_s, s)>rms)			return FALSE;
				if (rms_diff(_h, h)>((rms*4)/3))	return FALSE;
			}
		}
	}
	return TRUE;
}

u32	__stdcall rms_average	(lm_layer& lm, base_color_c& C)
{
	u32 x,y,_count=0;

	for (y=0; y<lm.height; y++)
	{
		for (x=0; x<lm.width; x++)
		{
			u32	offset	= y*lm.width+x;
			if (lm.marker[offset]>=254)	
			{
				base_color_c	cc;
				lm.surface		[offset]._get (cc);
				C.add			(cc);
				_count			++;
			}
		}
	}
	return	_count;
}


BOOL	compress_Zero		(lm_layer& lm, u32 rms)
{
	// Average color
	base_color_c	_c;
	u32				_count	= rms_average(lm,_c);

	if (0==_count)	{
		clMsg	("* ERROR: Lightmap not calculated (T:%d)");
		return	FALSE;
	} else		_c.scale(_count);

	// Compress if needed
	u8	_r	= u8_clr	(_c.rgb.x	); //.
	u8	_g	= u8_clr	(_c.rgb.y	);
	u8	_b	= u8_clr	(_c.rgb.z	);
	u8	_s	= u8_clr	(_c.sun		);
	u8	_h	= u8_clr	(_c.hemi	);
	if (rms_test(lm,_r,_g,_b,_s,_h,rms))
	{
		u32		c_x			= BORDER*2;
		u32		c_y			= BORDER*2;
		base_color ccc;		ccc._set(_c);
		lm.surface.assign	(c_x*c_y,ccc);
		lm.marker.assign	(c_x*c_y,255);
		lm.height			= 0;
		lm.width			= 0;
		return TRUE;
	}
	return FALSE;
}

BOOL	compress_RMS		(lm_layer& lm, u32 rms, u32& w, u32& h)
{
	// *** Try to bilinearly filter lightmap down and up
	w=0, h=0;
	if (lm.width>=2)	{
		w = lm.width/2;
		if (!rms_test(lm,w,lm.height,rms))	{
			// 3/4
			w = (lm.width*3)/4;
			if (!rms_test(lm,w,lm.height,rms))	w = 0;
		} else {
			// 1/4
			u32 nw = (lm.width*1)/4;
			if (rms_test(lm,nw,lm.height,rms))	w = nw;
		}
	}
	if (lm.height>=2)	{
		h = lm.height/2;
		if (!rms_test(lm,lm.width,h,rms))		{
			// 3/4
			h = (lm.height*3)/4;
			if (!rms_test(lm,lm.width,h,rms))		h = 0;
		} else {
			// 1/4
			u32 nh = (lm.height*1)/4;
			if (rms_test(lm,lm.width,nh,rms))		h = nh;
		}
	}
	if (w || h)	{
		if (0==w)	w = lm.width;
		if (0==h)	h = lm.height;
		//		clMsg	("* RMS: [%d,%d] => [%d,%d]",lm.width,lm.height,w,h);
		return TRUE;
	}
	return FALSE;
}


#ifndef DevCPU 
	#include "xrHardwareLight.h"
#endif

void CDeflector::Light(int th, CDB::COLLIDER* DB, base_lighting* LightsSelected, HASH& H)
{
	// Geometrical bounds
	Fbox bb;		bb.invalidate	();
	CTimer t;
	t.Start();
	try
	{
		for (u32 fid=0; fid<UVpolys.size(); fid++)
		{
			Face*	F		= UVpolys[fid].owner;
			for (int i=0; i<3; i++)	
				bb.modify(F->v[i]->P);
		}
		bb.getsphere(Sphere.P,Sphere.R);
	} 
	catch (...)
	{
		clMsg("* ERROR: CDeflector::Light - sphere calc");
	}
	
	u64 sphere_tick = t.GetElapsed_ticks();
	t.Start();

	// Convert lights to local form
	LightsSelected->select(inlc_global_data()->L_static(),Sphere.P,Sphere.R);
	 
	u64 Select_tick = t.GetElapsed_ticks();
	t.Start();

	// Calculate and fill borders
	L_Calculate			(th, DB,LightsSelected, H);

	u64 L_Calculate_tick = t.GetElapsed_ticks();
 	
	t.Start();

	if(_net_session && !_net_session->test_connection())
			 return;
   
	u64 NETSESSION_tick = t.GetElapsed_ticks();
	//Msg("_net_session: %llu", t.GetElapsed_ticks());

#ifndef DevCPU 
	if (xrHardwareLight::Get().IsEnabled())
 
	{
#ifdef OLD_METHOD_GPU_COMPUTE
		LightEnd(th, DB, LightsSelected, H);
#endif
	}
	else 
	{
		LightEnd(th, DB, LightsSelected, H);
	}
#else 
	
	t.Start();

	for (u32 ref = 254; ref > 0; ref--)
		if (!ApplyBorders(layer, ref))
			break;

	u64 ApplyBorders_ticks = t.GetElapsed_ticks();
	//Msg("ApplyBorders: %llu", t.GetElapsed_ticks());


	t.Start();
	// Compression

	try
	{
		u32	w, h;
		if (compress_Zero(layer, rms_zero))
		{
			return;		// already with borders
		}
		else
			if (compress_RMS(layer, rms_shrink, w, h))
			{
				// Reacalculate lightmap at lower resolution
				layer.create(w, h);
				L_Calculate(th, DB, LightsSelected, H, true);
				if (_net_session && !_net_session->test_connection())
					return;
			}

	}
	catch (...)
	{
		clMsg("* ERROR: CDeflector::Light - Compression");
	}

	u64 Compression = t.GetElapsed_ticks();
	//Msg("Compression: %llu", t.GetElapsed_ticks());

	t.Start();
	// Expand with borders
	try
	{
		if (layer.width == 1)
		{
			// Horizontal ZERO - vertical line
			lm_layer		T;
			T.create(2 * BORDER, layer.height + 2 * BORDER);

			// Transfer
			for (u32 y = 0; y < T.height; y++)
			{
				int			py = int(y) - BORDER;
				clamp(py, 0, int(layer.height - 1));
				base_color	C = layer.surface[py];
				T.surface[y * 2 + 0] = C;
				T.marker[y * 2 + 0] = 255;
				T.surface[y * 2 + 1] = C;
				T.marker[y * 2 + 1] = 255;
			}

			// Exchange
			T.width = 0;
			T.height = layer.height;
			layer = T;
		}
		else if (layer.height == 1)
		{
			// Vertical ZERO - horizontal line
			lm_layer		T;
			T.create(layer.width + 2 * BORDER, 2 * BORDER);

			// Transfer
			for (u32 x = 0; x < T.width; x++)
			{
				int			px = int(x) - BORDER;
				clamp(px, 0, int(layer.width - 1));
				base_color	C = layer.surface[px];
				T.surface[0 * T.width + x] = C;
				T.marker[0 * T.width + x] = 255;
				T.surface[1 * T.width + x] = C;
				T.marker[1 * T.width + x] = 255;
			}

			// Exchange
			T.width = layer.width;
			T.height = 0;
			layer = T;
		}
		else
		{
			// Generic blit
			lm_layer		lm_old = layer;
			lm_layer		lm_new;
			lm_new.create(lm_old.width + 2 * BORDER, lm_old.height + 2 * BORDER);
			lblit(lm_new, lm_old, BORDER, BORDER, 255 - BORDER);
			layer = lm_new;
			ApplyBorders(layer, 254);
			ApplyBorders(layer, 253);
			ApplyBorders(layer, 252);
			ApplyBorders(layer, 251);
			for (u32 ref = 250; ref > 0; ref--) if (!ApplyBorders(layer, ref)) break;
			layer.width = lm_old.width;
			layer.height = lm_old.height;
		}
	}
	catch (...)
	{
		clMsg("* ERROR: CDeflector::Light - BorderExpansion");
	}

	u64 BorderExpansion = t.GetElapsed_ticks();
	//Msg("BorderExpansion: %llu", t.GetElapsed_ticks());

	//string256 tmp;
	//sprintf(tmp, "Sphere[%u], Select[%u], LCAlculate[%u], NETSESSION[%u], ApplyBorders[%u],  Compression[%u], Borders[%u]", sphere_tick, Select_tick, L_Calculate_tick, ApplyBorders_ticks, Compression, BorderExpansion);

	//Msg(tmp);
#endif
	
}

void CDeflector::LightEnd(int th, CDB::COLLIDER* DB, base_lighting* LightsSelected, HASH& H)
{
 
#ifndef DevCPU
#ifndef OLD_METHOD_GPU_COMPUTE  
	if (xrHardwareLight::Get().IsEnabled())	
	{	
		LightsSelected->select(inlc_global_data()->L_static(),Sphere.P,Sphere.R);
	
		// UV & HASH
 		Fbox2			bounds;
		Bounds_Summary	(bounds);
		H.initialize	(bounds,(u32)UVpolys.size());

		for (u32 fid=0; fid<UVpolys.size(); fid++)
		{
			UVtri* T	= &(UVpolys[fid]);
			Bounds		(fid,bounds);
			H.add		(bounds,T);
		}
	}
#endif
#endif
 	for (u32 ref = 254; ref > 0; ref--)
	if (!ApplyBorders(layer, ref))
		break;

	// Compression
	try
	{
		u32	w, h;
		if (compress_Zero(layer, rms_zero))
		{
			return;		// already with borders
		}
		else
		if (compress_RMS(layer, rms_shrink, w, h))
		{
			// Reacalculate lightmap at lower resolution
			layer.create(w, h);
			L_Calculate(th, DB, LightsSelected, H, true);
			if (_net_session && !_net_session->test_connection())
				return;
		}
		
	}
	catch (...)
	{
		clMsg("* ERROR: CDeflector::Light - Compression");
	}
	
	// Expand with borders
	try
	{
		if (layer.width == 1)
		{
			// Horizontal ZERO - vertical line
			lm_layer		T;
			T.create(2 * BORDER, layer.height + 2 * BORDER);

			// Transfer
			for (u32 y = 0; y < T.height; y++)
			{
				int			py = int(y) - BORDER;
				clamp(py, 0, int(layer.height - 1));
				base_color	C = layer.surface[py];
				T.surface[y * 2 + 0] = C;
				T.marker[y * 2 + 0] = 255;
				T.surface[y * 2 + 1] = C;
				T.marker[y * 2 + 1] = 255;
			}

			// Exchange
			T.width = 0;
			T.height = layer.height;
			layer = T;
		}
		else if (layer.height == 1)
		{
			// Vertical ZERO - horizontal line
			lm_layer		T;
			T.create(layer.width + 2 * BORDER, 2 * BORDER);

			// Transfer
			for (u32 x = 0; x < T.width; x++)
			{
				int			px = int(x) - BORDER;
				clamp(px, 0, int(layer.width - 1));
				base_color	C = layer.surface[px];
				T.surface[0 * T.width + x] = C;
				T.marker[0 * T.width + x] = 255;
				T.surface[1 * T.width + x] = C;
				T.marker[1 * T.width + x] = 255;
			}

			// Exchange
			T.width = layer.width;
			T.height = 0;
			layer = T;
		}
		else 
		{
			// Generic blit
			lm_layer		lm_old = layer;
			lm_layer		lm_new;
			lm_new.create(lm_old.width + 2 * BORDER, lm_old.height + 2 * BORDER);
			lblit(lm_new, lm_old, BORDER, BORDER, 255 - BORDER);
			layer = lm_new;
			ApplyBorders(layer, 254);
			ApplyBorders(layer, 253);
			ApplyBorders(layer, 252);
			ApplyBorders(layer, 251);
			for (u32 ref = 250; ref > 0; ref--) if (!ApplyBorders(layer, ref)) break;
			layer.width = lm_old.width;
			layer.height = lm_old.height;
		}
	}
	catch (...)
	{
		clMsg("* ERROR: CDeflector::Light - BorderExpansion");
	}
}