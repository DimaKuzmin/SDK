#include "stdafx.h"
#include "compiler.h"
#include "guid_generator.h"

IC BYTE	compress(float c, int max_value)
{
	int	cover = iFloor(c*float(max_value)+.5f);
	clamp(cover,0,max_value);
	return BYTE(cover);
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
xr_vector<NodeCompressed11>	compressed_nodes_v11;
 
// c++ 17 Style
template<typename NodePacked>
class CNodeRenumberer 
{
public:
	CNodeRenumberer( xr_vector<NodePacked>& nodes, xr_vector<u32>& sorted, xr_vector<u32>& renumbering ) :
		m_nodes(nodes),
		m_sorted(sorted),
		m_renumbering(renumbering)
	{
		const std::size_t N = m_nodes.size();
		m_sorted.resize(N);
		m_renumbering.resize(N);

		// Инициализация индексов
		for (std::size_t i = 0; i < N; ++i)
			m_sorted[i] = static_cast<u32>(i);

		// Сортировка по координате xz
		std::stable_sort(m_sorted.begin(), m_sorted.end(), SSortNodesPredicate{ m_nodes });

		// Построение таблицы перенумерации
		for (std::size_t i = 0; i < N; ++i)
			m_renumbering[m_sorted[i]] = static_cast<u32>(i);

		// Обновление ссылок
		for (std::size_t i = 0; i < N; ++i) 
		{
			for (u8 j = 0; j < 4; ++j) {
				u32 vertex_id = m_nodes[i].link(j);
				if (vertex_id < N)
					m_nodes[i].link(j, m_renumbering[vertex_id]);
			}
		}

		// Переставить сами узлы в отсортированный порядок
		std::stable_sort(m_nodes.begin(), m_nodes.end(), SSortNodesPredicate{});
	}

	// Копирование запрещено
	CNodeRenumberer(const CNodeRenumberer&) = delete;
	CNodeRenumberer& operator=(const CNodeRenumberer&) = delete;

private:
	struct SSortNodesPredicate 
	{
		const xr_vector<NodePacked>* nodes_ptr = nullptr;

		// Сравнение по координате xz для самих узлов
		bool operator()(const NodePacked& a, const NodePacked& b) const {
			return a.p.xz() < b.p.xz();
		}

		// Сравнение по координате xz по id
		bool operator()(u32 id0, u32 id1) const {
			return (*nodes_ptr)[id0].p.xz() < (*nodes_ptr)[id1].p.xz();
		}

		// Позволяет использовать структуру и для узлов, и для индексов
		SSortNodesPredicate() = default;
		SSortNodesPredicate(const xr_vector<NodePacked>& nodes) : nodes_ptr(&nodes) {}
	};

	xr_vector<NodePacked>& m_nodes;
	xr_vector<u32>& m_sorted;
	xr_vector<u32>& m_renumbering;
};

// Pack nodes

template <typename T>
void CNodePositionCompressor(T& Pdest, Fvector& Psrc, hdrNODES& H)
{
	float sp = 1 / g_params.fPatchSize;
	int row_length = iFloor((H.aabb.max.z - H.aabb.min.z) / H.size + EPS_L + 1.5f);

	int pxz = iFloor((Psrc.x - H.aabb.min.x) * sp + EPS_L + .5f) * row_length + iFloor((Psrc.z - H.aabb.min.z) * sp + EPS_L + .5f);
	int py = iFloor(65535.f * (Psrc.y - H.aabb.min.y) / (H.size_y) + EPS_L);

	if (pxz > u32(1 << MAX_NODE_BIT_COUNT))
	{
		xrADD_ERRORED_NODE(pxz);
	}

	//VERIFY	(pxz < (1 << MAX_NODE_BIT_COUNT) - 1);
	Pdest.xz(pxz);
	clamp(py, 0, 65535);
	Pdest.y(u16(py));
}


template<typename T>
void	CompressNodeNew(T& Dest, vertex& Src, hdrNODES& H)
{
	// Compress plane (normal)
	Dest.plane = pvCompress(Src.Plane.n);

	// Compress position
	CNodePositionCompressor(Dest.p, Src.Pos, H);

	// Light & Cover
	Dest.light(15);
	for (u8 L = 0; L < 4; ++L)
		Dest.link(L, Src.n[L]);

	Dest.high.cover0 = compress(Src.high_cover[0], 15);
	Dest.high.cover1 = compress(Src.high_cover[1], 15);
	Dest.high.cover2 = compress(Src.high_cover[2], 15);
	Dest.high.cover3 = compress(Src.high_cover[3], 15);
	Dest.low.cover0 = compress(Src.low_cover[0], 15);
	Dest.low.cover1 = compress(Src.low_cover[1], 15);
	Dest.low.cover2 = compress(Src.low_cover[2], 15);
	Dest.low.cover3 = compress(Src.low_cover[3], 15);
}



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
	
	int AIMapVersion = gCompilerMode.AI_Map_NoLimits ? 11 : 10; 	
	
	hdrNODES		H;
	H.version		= AIMapVersion;
	H.count			= g_nodes.size();
	H.size			= g_params.fPatchSize;
	H.size_y		= CalculateHeight(H.aabb);
	H.guid			= generate_guid();
	fs->w			(&H,sizeof(H));

	// All nodes
	Status			("Saving nodes...");
  	for (u32 i = 0; i < g_nodes.size(); ++i)
	{
		vertex& N = g_nodes[i];
		if (gCompilerMode.AI_Map_NoLimits)
		{
			NodeCompressed11	NC;
			CompressNodeNew(NC, N, H);
			compressed_nodes_v11.push_back(NC);
		}
		else
		{
			NodeCompressed	NC;
			CompressNodeNew(NC, N, H);
			compressed_nodes.push_back(NC);
		}
 	}
	 
	int n_e = errored_nodes;
 
	clMsg("nodes Size[%u], memory[%u] KB, count_error: %u",  compressed_nodes.size(), (compressed_nodes.size() / 1024) * sizeof(NodeCompressed), g_nodes.size() );
	clMsg("--- [SE7KILLS] VersionOf AiMAP: %d", AIMapVersion);
 
	xr_vector<u32>	sorted;
	xr_vector<u32>	renumbering;
	
	if (gCompilerMode.AI_Map_NoLimits)
	{
		CNodeRenumberer data (compressed_nodes_v11, sorted, renumbering);

		// Write Packed
		for (u32 i = 0; i < g_nodes.size(); ++i)
		{
			fs->w(&compressed_nodes_v11[i], sizeof(NodeCompressed11));
			Progress(float(i) / float(g_nodes.size()));
		}
	}
	else
	{
		CNodeRenumberer	A(compressed_nodes, sorted, renumbering);
		for (u32 i = 0; i < g_nodes.size(); ++i)
		{
			fs->w(&compressed_nodes[i], sizeof(NodeCompressed));
			Progress(float(i) / float(g_nodes.size()));
		}
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
