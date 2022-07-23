#include "stdafx.h"

#include "base_face.h"
#include "serialize.h"

base_Face::base_Face()
{
	basis_tangent[0].set	(0,0,0);
	basis_tangent[1].set	(0,0,0);
	basis_tangent[2].set	(0,0,0);
	basis_binormal[0].set	(0,0,0);
	basis_binormal[1].set	(0,0,0);
	basis_binormal[2].set	(0,0,0);
}
base_Face::~base_Face()		{};

base_Vertex::~base_Vertex() {};


/*

	base_basis				basis_tangent		[3];
	base_basis				basis_binormal		[3];
	u16						dwMaterial;			// index of material
	u16						dwMaterialGame;		// unique-id of game material (must persist up to game-CForm saving)

		struct					{
		u16					bSplitted			:		1;
		u16					bProcessed			:		1;
		u16					bOpaque				:		1;	// For ray-tracing speedup
		u16					bLocked				:		1;	// For tesselation
	}						flags;
*/
void		base_Face::	read	( INetReader	&r )
{
	r_pod<base_basis[3]>(r, basis_tangent );
	r_pod<base_basis[3]>(r, basis_binormal );
	dwMaterial = r.r_u16();
	dwMaterialGame = r.r_u16();	
	r_pod( r, flags );
	
}
void		base_Face::	write	( IWriter	&w )const
{
	w_pod<base_basis[3]>( w, basis_tangent );
	w_pod<base_basis[3]>( w, basis_binormal );
	w.w_u16( dwMaterial );
	w.w_u16( dwMaterialGame );	
	w_pod( w, flags );
}

/*	Fvector					P;
	Fvector					N;
	base_color				C;			// all_lighting info
	int						handle;		// used in mesh-processing/optimization/conversion
*/

void	base_Vertex	::	read	(INetReader	&r )
{
	r.r_fvector3(P );
	r.r_fvector3( N );
	r_pod(r,C);		
	handle = r.r_s32();
}
void	base_Vertex::		write	(IWriter	&w )const
{
	 w.w_fvector3( P );
	 w.w_fvector3( N );
	 w_pod( w, C );		
	 w.w_s32( handle );
}

void base_Vertex::read_Reader(IReader& r)
{
	r.r_fvector3(P);
	r.r_fvector3(N);
	r_pod_reader(r, C);
	handle = r.r_s32();
}

void base_Vertex::write_Reader(IWriter& w)
{
	w.w_fvector3(P);
	w.w_fvector3(N);
	w_pod_writer(w, C);
	w.w_s32(handle);
}

void base_Vertex::read_LTX(CInifile* file, LPCSTR sec)
{
}

void base_Vertex::write_LTX(CInifile* file, LPCSTR sec)
{
}

void base_Face::read_Reader(IReader& r)
{
	r_pod_reader<base_basis[3]>(r, basis_tangent);
	r_pod_reader<base_basis[3]>(r, basis_binormal);
	dwMaterial = r.r_u16();
	dwMaterialGame = r.r_u16();
	r_pod_reader(r, flags);
}

void base_Face::write_Reader(IWriter& w)
{
	w_pod_writer<base_basis[3]>(w, basis_tangent);
	w_pod_writer<base_basis[3]>(w, basis_binormal);
	w.w_u16(dwMaterial);
	w.w_u16(dwMaterialGame);
	w_pod_writer(w, flags);
}

void base_Face::read_LTX(CInifile* file, LPCSTR sec)
{


}

void base_Face::write_LTX(CInifile* file, LPCSTR sec)
{
	basis_tangent->save_ltx(file, sec, "tangent");
	basis_binormal->save_ltx(file, sec, "binormal");
	file->w_u16(sec, "material", dwMaterial);
	file->w_u16(sec, "materialGame", dwMaterialGame);
	file->w_bool(sec, "bLocked", flags.bLocked);
	file->w_bool(sec, "bOpaque", flags.bOpaque);
	file->w_bool(sec, "bProcessed", flags.bProcessed);
	file->w_bool(sec, "bSplitted", flags.bSplitted);
}
