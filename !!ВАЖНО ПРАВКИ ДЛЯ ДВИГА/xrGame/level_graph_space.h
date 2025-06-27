////////////////////////////////////////////////////////////////////////////
//	Module 		: level_graph_space.h
//	Created 	: 02.10.2001
//  Modified 	: 08.12.2004
//	Author		: Dmitriy Iassenev
//	Description : Level graph space
////////////////////////////////////////////////////////////////////////////

#pragma once

namespace LevelGraph {
	class CHeader : private hdrNODES {
	private:
		friend class CRenumbererConverter;

	public:
		ICF	u32				version					() const;
		ICF	u32				vertex_count			() const;
		ICF	float			cell_size				() const;
		ICF	float			factor_y				() const;
		ICF	const Fbox		&box					() const;
		ICF const xrGUID	&guid					() const;
	};


	struct NodePos
	{
		u32 XZ_value;
		u16 Y_value;
	public:
		ICF	void xz(u32 value) { XZ_value = value; }
		ICF	void y(u16 value) { Y_value = value; }


		ICF	u32	x(u32 row) const
		{
			return			(xz() / row);
		}
		ICF	u32	z(u32 row) const
		{
			return			(xz() % row);
		}

		ICF	u32	xz() const
		{
			return			XZ_value;
		}

		ICF	u32	y() const {
			return			Y_value;
		}

		friend class	CLevelGraph;
	};

	typedef NodePos CPosition;

	class CVertex
	{


		SCover				high;
		SCover				low;

		u16					plane_value;
		u32					link_value[4];

		CPosition			position_value;
	public:

		ICF	u32				link(int i) const;
		ICF	u16				high_cover(u8 index) const;
		ICF	u16				low_cover(u8 index) const;
		ICF	u16				plane() const;

		ICF	const CPosition& position() const;
		ICF	bool			operator<				(const LevelGraph::CVertex& vertex) const;
		ICF	bool			operator>				(const LevelGraph::CVertex& vertex) const;
		ICF	bool			operator==				(const LevelGraph::CVertex& vertex) const;

		friend class CLevelGraph;
	};

	struct SSegment {
		Fvector v1;
		Fvector v2;
	};

	struct SContour : public SSegment {
		Fvector v3;
		Fvector v4;
	};
};
