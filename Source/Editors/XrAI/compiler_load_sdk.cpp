#include "StdAfx.h"
#include "compiler.h"

template <typename T>
void CNodePositionCompressor_SDK(T& Pdest, Fvector& Psrc, hdrNODES& H)
{
	float sp = 1 / g_params.fPatchSize;
	int row_length = iFloor((H.aabb.max.z - H.aabb.min.z) / H.size + EPS_L + 1.5f);

	int pxz = iFloor((Psrc.x - H.aabb.min.x) * sp + EPS_L + .5f) * row_length + iFloor((Psrc.z - H.aabb.min.z) * sp + EPS_L + .5f);
	int py = iFloor(65535.f * (Psrc.y - H.aabb.min.y) / (H.size_y) + EPS_L);

	Pdest.xz(pxz);
	clamp(py, 0, 65535);
	Pdest.y(u16(py));
}


void CNodePositionConverterSDK(const SNodePositionOld& Psrc, hdrNODES& m_header, LevelGraph::CPosition& np)
{
	Fvector		Pdest;
	Pdest.x = float(Psrc.x) * m_header.size;
	Pdest.y = (float(Psrc.y) / 65535) * m_header.size_y + m_header.aabb.min.y;
	Pdest.z = float(Psrc.z) * m_header.size;
	CNodePositionCompressor_SDK(np, Pdest, m_header);
	np.y(Psrc.y);
}

IC	const Fvector vertex_position(const CLevelGraph::CPosition& Psrc, const Fbox& bb, const SAIParams& params)
{
	Fvector				Pdest;
	int	x, z, row_length;
	row_length = iFloor((bb.max.z - bb.min.z) / params.fPatchSize + EPS_L + 1.5f);
	x = Psrc.xz() / row_length;
	z = Psrc.xz() % row_length;
	Pdest.x = float(x) * params.fPatchSize + bb.min.x;
	Pdest.y = (float(Psrc.y()) / 65535) * (bb.max.y - bb.min.y) + bb.min.y;
	Pdest.z = float(z) * params.fPatchSize + bb.min.z;
	return				(Pdest);
}

void compiler_load_sdk_nodes(LPCSTR name)
{
	// Load initial map from the Level Editor
	{
		string_path			file_name;
		strconcat(sizeof(file_name), file_name, name, "build.aimap");
		IReader* F = FS.r_open(file_name);
		R_ASSERT2(F, file_name);

		R_ASSERT(F->open_chunk(E_AIMAP_CHUNK_VERSION));
		R_ASSERT(F->r_u16() == E_AIMAP_VERSION);

		R_ASSERT(F->open_chunk(E_AIMAP_CHUNK_BOX));
		F->r(&LevelBB, sizeof(LevelBB));

		R_ASSERT(F->open_chunk(E_AIMAP_CHUNK_PARAMS));
		F->r(&g_params, sizeof(g_params));

		R_ASSERT(F->open_chunk(E_AIMAP_CHUNK_NODES));
		u32					N = F->r_u32();

		R_ASSERT2(N < ((u32(1) << u32(MAX_NODE_BIT_COUNT)) - 2), "Too many nodes!");

		Msg("Load Nodes Size: %d", N);
		g_nodes.resize(N);

		hdrNODES			H;
		H.version = XRAI_CURRENT_VERSION;
		H.count = N + 1;
		H.size = g_params.fPatchSize;
		H.size_y = 1.f;
		H.aabb = LevelBB;

		for (u32 i = 0; i < N; i++)
		{
			u16 				pl;
			SNodePositionOld 	_np;
		
			LevelGraph::CPosition 		np;

			for (int j = 0; j < 4; ++j)
			{
				u32 id = F->r_u32();
				g_nodes[i].n[j] = id;
			}

			pl = F->r_u16();
			pvDecompress(g_nodes[i].Plane.n, pl);
			F->r(&_np, sizeof(_np));

			CNodePositionConverterSDK(_np, H, np);
			g_nodes[i].Pos = vertex_position(np, LevelBB, g_params);
			g_nodes[i].Plane.build(g_nodes[i].Pos, g_nodes[i].Plane.n);

		}

		Msg("Level Nodes %d", g_nodes.size());
		Msg("Level Min BB [%f][%f][%f]", LevelBB.min.x, LevelBB.min.y, LevelBB.min.z);
		Msg("Level Max BB [%f][%f][%f]", LevelBB.max.x, LevelBB.max.y, LevelBB.max.z);

		F->close();
	}
}

