#include "stdafx.h"
#include "compiler.h"
#include "guid_generator.h"

IC BYTE	compress(float c, int max_value)
{
	int	cover = iFloor(c*float(max_value)+.5f);
	clamp(cover,0,max_value);
	return BYTE(cover);
}

struct CNodeCompressed {
	IC	void	compress_node(NodeCompressed& Dest, vertex& Src);
};

IC void	CNodeCompressed::compress_node(NodeCompressed& Dest, vertex& Src)
{
	Dest.light	(15);//compress(Src.LightLevel,15));
	for	(u8 L=0; L<4; ++L)
		Dest.link(L,Src.n[L]);
//	for	(u32 L=0; L<4; ++L)
//		if ((Src.n[L] < g_nodes.size()) && (Dest.link(L) != Src.n[L])) {
//			Dest.link(L,Src.n[L]);
//			Dest.link(L);
//		}
}

void	Compress	(NodeCompressed& Dest, vertex& Src, hdrNODES& H)
{
	// Compress plane (normal)
	Dest.plane	= pvCompress	(Src.Plane.n);
	
	// Compress position
	CNodePositionCompressor(Dest.p,Src.Pos,H);
//	CompressPos	(Dest.p1,Src.P1,H);
	
	// Sector
	// R_ASSERT(Src.sector<=255);
	// Dest.sector = BYTE(Src.sector);

	// Light & Cover
	CNodeCompressed().compress_node(Dest,Src);
//	Dest.cover[0]	= CompressCover(Src.cover[0]);
//	Dest.cover[1]	= CompressCover(Src.cover[1]);
//	Dest.cover[2]	= CompressCover(Src.cover[2]);
//	Dest.cover[3]	= CompressCover(Src.cover[3]);
	Dest.high.cover0= compress(Src.high_cover[0],15);
	Dest.high.cover1= compress(Src.high_cover[1],15);
	Dest.high.cover2= compress(Src.high_cover[2],15);
	Dest.high.cover3= compress(Src.high_cover[3],15);
	Dest.low.cover0	= compress(Src.low_cover[0],15);
	Dest.low.cover1	= compress(Src.low_cover[1],15);
	Dest.low.cover2	= compress(Src.low_cover[2],15);
	Dest.low.cover3	= compress(Src.low_cover[3],15);
//	Msg				("[%.3f -> %d][%.3f -> %d][%.3f -> %d][%.3f -> %d]",
//		Src.cover[0],Dest.cover0,
//		Src.cover[1],Dest.cover1,
//		Src.cover[2],Dest.cover2,
//		Src.cover[3],Dest.cover3
//		);

	// Compress links
//	R_ASSERT	(Src.neighbours.size()<64);
//	Dest.links	= BYTE(Src.neighbours.size());
}

float	CalculateHeight(Fbox& BB)
{
	// All nodes
	BB.invalidate();

	for (u32 i=0; i<g_nodes.size(); i++)
	{
		vertex&	N	= g_nodes[i];
		BB.modify	(N.Pos);
	}
	return BB.max.y-BB.min.y+EPS_L;
}

xr_vector<NodeCompressed>	compressed_nodes;

class CNodeRenumberer {
	IC	bool operator=	(const CNodeRenumberer&)
	{
	}

	struct SSortNodesPredicate {

		IC	bool	operator()			(const NodeCompressed &vertex0, const NodeCompressed &vertex1) const
		{
			return		(vertex0.p.xz() < vertex1.p.xz());
		}


		IC	bool	operator()			(u32 vertex_id0, u32 vertex_id1) const
		{
			return		(compressed_nodes[vertex_id0].p.xz() < compressed_nodes[vertex_id1].p.xz());
		}
	};

	xr_vector<NodeCompressed>	&m_nodes;
	xr_vector<u32>				&m_sorted;
	xr_vector<u32>				&m_renumbering;

public:
					CNodeRenumberer(
						xr_vector<NodeCompressed>	&nodes, 
						xr_vector<u32>				&sorted,
						xr_vector<u32>				&renumbering
					) :
						m_nodes(nodes),
						m_sorted(sorted),
						m_renumbering(renumbering)
	{
		u32					N = (u32)m_nodes.size();
		m_sorted.resize		(N);
		m_renumbering.resize(N);

		for (u32 i=0; i<N; ++i)
			m_sorted[i]		= i;

		std::stable_sort	(m_sorted.begin(),m_sorted.end(),SSortNodesPredicate());

		for (u32 i=0; i<N; ++i)
			m_renumbering	[m_sorted[i]] = i;

		for (u32 i=0; i<N; ++i) {
			for (u32 j=0; j<4; ++j) {
				u32			vertex_id = m_nodes[i].link(u8(j));
				if (vertex_id >= N)
					continue;
				m_nodes[i].link(u8(j),m_renumbering[vertex_id]);
			}
		}

		std::stable_sort	(m_nodes.begin(),m_nodes.end(),SSortNodesPredicate());
	}
};

int errored_nodes = 0;
int E_MAX_PXZ = 0;

void xrSaveNodes(LPCSTR N, LPCSTR out_name)
{
	Msg				("NS: %d, CNS: %d, ratio: %f%%",sizeof(vertex),sizeof(CLevelGraph::CVertex),100*float(sizeof(CLevelGraph::CVertex))/float(sizeof(vertex)));

	Msg				("Renumbering nodes...");

	string_path		fName; 
	strconcat		(sizeof(fName),fName,N,out_name);

	IWriter			*fs = FS.w_open(fName);

	// Header
	Status			("Saving header...");
	hdrNODES		H;
	H.version		= XRAI_CURRENT_VERSION;
	H.count			= g_nodes.size();
	H.size			= g_params.fPatchSize;
	H.size_y		= CalculateHeight(H.aabb);
	H.guid			= generate_guid();
	fs->w			(&H,sizeof(H));
	
//	fs->w_u32		(g_covers_palette.size());
//	for (u32 j=0; j<g_covers_palette.size(); ++j)
//		fs->w		(&g_covers_palette[j],sizeof(g_covers_palette[j]));

	// All nodes
	Status			("Saving nodes...");
	int count = 0;
	int MAX_PX = 0;

	int MAX_X = 0;
	int MAX_Y = 0;
	int MAX_Z = 0;
	int ROW_SIZE = 0;

	Msg("Min[%f][%f][%f]", H.aabb.min.x, H.aabb.min.y, H.aabb.min.z);
	Msg("Max[%f][%f][%f]", H.aabb.max.x, H.aabb.max.y, H.aabb.max.z);
 
	string_path path;
	FS.update_path(path, "$app_data_root$", "logs\\xrAI_NodesSave_Errors.log");
		
	IWriter* w = FS.w_open(path);

	for (u32 i=0; i<g_nodes.size(); ++i) 
	{
		vertex			&N	= g_nodes[i];
		NodeCompressed	NC;
		Compress		(NC,N,H);
		compressed_nodes.push_back(NC);

		if (NC.p.y() > MAX_Y)
		{
			Fvector Psrc = N.Pos;
			//int iFloorPosY = iFloor(65535.f * (Psrc.y - H.aabb.min.y) / (H.size_y) + EPS_L);
			MAX_Y = NC.p.y(); //iFloorPosY;

		}
	  
		if (NC.p.xz() > MAX_PX)
		{
			MAX_PX = NC.p.xz();

			float sp = 1 / g_params.fPatchSize;
			int row_length = iFloor((H.aabb.max.z - H.aabb.min.z) / H.size + EPS_L + 1.5f);

			Fvector Psrc = N.Pos;

			//int pxz = iFloor((Psrc.x - H.aabb.min.x) * sp + EPS_L + .5f) * row_length + iFloor((Psrc.z - H.aabb.min.z) * sp + EPS_L + .5f);

			int iFloorPosX =  ( (Psrc.x - H.aabb.min.x) * sp + EPS_L + .5f );
			int iFloorPosZ = iFloor((Psrc.z - H.aabb.min.z) * sp + EPS_L + .5f);
			MAX_X = iFloorPosX;
			MAX_Z = iFloorPosZ;

			ROW_SIZE = row_length;
		}
	
		/*
		// Unpack For Check Validate
		int m_row_length = iFloor((H.aabb.max.z - H.aabb.min.z) / H.size + EPS_L + 1.5f);
				
		float x, z;
		x = NC.p.xz() / m_row_length;
		z = NC.p.xz() % m_row_length;
		x = float(x) * H.size + H.aabb.min.x;
		z = float(z) * H.size + H.aabb.min.z;
		
		float py = float(  (NC.p.y()  / 65535.f) * H.size_y + H.aabb.min.y);


#ifdef _USE_NODE_POSITION_11
		int max_px = 0xffffffff;
#else 
		int max_px = 0x00ffffff;
#endif

		if (NC.p.xz() > max_px)
		{
			string64 tmp;
			sprintf(tmp, "Check: Node XZ: [%d], POS[%f][%f][%f] = value: %d, UX: %f, UY: %f, UZ: %f", i, VPUSH(N.Pos), NC.p.xz(), x, py, z);
			w->w_string(tmp);
		}


		if (NC.p.xz() > max_px)	  
		{
			count++;
			if (NC.p.xz() > MAX_PX)
			{
				MAX_PX = NC.p.xz();
				clMsg("NEW ROW[%d] Real x[%f], y[%f], z[%f] unpacked x[%f], y[%f], z[%f], compressed [%d] ", m_row_length, N.Pos.x, N.Pos.y, N.Pos.z, x, py, z, NC.p.xz());
			}
		}

		//if (i % 1024 == 0)
		//	clMsg("NEW ROW[%d] Real x[%f], y[%f], z[%f] unpacked x[%f], y[%f], z[%f], compressed [%d] ", m_row_length, N.Pos.x, N.Pos.y, N.Pos.z, x, py, z, NC.p.xz());
		*/
	}

	FS.w_close(w);
 
	int n_e = errored_nodes;
 
	clMsg("nodes Size[%u], memory[%u] KB, count_error: %u", 
		compressed_nodes.size(), (compressed_nodes.size() / 1024) * sizeof(NodeCompressed),
		count
	);

	clMsg(" MAXPXZ[%u], MAX_X[%u], MAX_Z[%u], MAX_Y[%u] ROW[%u]",
		MAX_PX, MAX_X, MAX_Z, MAX_Y, ROW_SIZE
		);
 
	xr_vector<u32>	sorted;
	xr_vector<u32>	renumbering;
	CNodeRenumberer	A(compressed_nodes,sorted,renumbering);
	for (u32 i=0; i < g_nodes.size(); ++i) 
	{
		fs->w			(&compressed_nodes[i],sizeof(NodeCompressed));
		Progress		(float(i)/float(g_nodes.size()));
	}
	// Stats
	u32	SizeTotal	= fs->tell();
	Msg				("%dK saved",SizeTotal/1024);

	FS.w_close		(fs);
}

void xrADD_ERRORED_NODE(int pxz)
{
	errored_nodes++;
	if (E_MAX_PXZ < pxz)
	E_MAX_PXZ = pxz;
}
