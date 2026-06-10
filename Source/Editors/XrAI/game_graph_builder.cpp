////////////////////////////////////////////////////////////////////////////
//	Module 		: game_graph_builder.cpp
//	Created 	: 14.12.2005
//  Modified 	: 14.12.2005
//	Author		: Dmitriy Iassenev
//	Description : Game graph builder
////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "game_graph_builder.h"
#include "level_graph.h"
#include "graph_abstract.h"
#include "xrMessages.h"
#include "xrServer_Objects_ALife.h"
#include "factory_api.h"
#include "game_level_cross_table.h" 
#include "guid_generator.h"
#include "graph_engine.h"

CGameGraphBuilder::CGameGraphBuilder		()
{
	m_level_graph			= 0;
	m_graph					= 0;
	m_cross_table			= 0;
}

CGameGraphBuilder::~CGameGraphBuilder		()
{
	Msg						("Freeing resources");

	xr_delete				(m_level_graph);
	xr_delete				(m_graph);
	xr_delete				(m_cross_table);
}

void CGameGraphBuilder::create_graph		(const float &start, const float &amount)
{
	m_graph					= xr_new<graph_type>();
	m_graph_guid			= generate_guid();
}

void CGameGraphBuilder::load_level_graph	(const float &start, const float &amount)
{
 	m_level_graph			= xr_new<CLevelGraph>(*m_level_name);	
	Msg("%d nodes loaded", level_graph().header().vertex_count());
}

void CGameGraphBuilder::load_graph_point	(NET_Packet &net_packet)
{
	string256				section_id;
	u16						id;
	net_packet.r_begin		(id);
	R_ASSERT				(M_SPAWN == id);
	net_packet.r_stringZ	(section_id);

	CSE_Abstract			*entity = F_entity_Create(section_id);
	if (!entity) {
		Msg					("Cannot create entity from section %s, skipping", section_id);
		return;
	}

	CSE_ALifeGraphPoint		*graph_point = smart_cast<CSE_ALifeGraphPoint*>(entity);
	if (!graph_point) {
		F_entity_Destroy	(entity);
		return;
	}

	entity->Spawn_Read		(net_packet);

	vertex_type				vertex;
	vertex.tLocalPoint		= graph_point->o_Position;
	// check for duplicate graph point positions
	{
		graph_type::const_vertex_iterator	I = graph().vertices().begin();
		graph_type::const_vertex_iterator	E = graph().vertices().end();
		for ( ; I != E; ++I) 
		{
			if ((*I).second->data().tLocalPoint.distance_to_sqr(vertex.tLocalPoint) < EPS_L)
			{
				Msg			("! removing graph point [%s][%f][%f][%f] because it is too close to the another graph point",
					entity->name_replace(),VPUSH(entity->o_Position));
				F_entity_Destroy(entity);
				return;
			}
		}
	}

	vertex.tGlobalPoint		= graph_point->o_Position;
	vertex.tNodeID			= level_graph().valid_vertex_position(vertex.tLocalPoint) ? level_graph().vertex_id(vertex.tLocalPoint) : u32(-1);
	
	if (!level_graph().valid_vertex_id(vertex.tNodeID))
	{
 		Msg					("! removing graph point [%s][%f][%f][%f] because it is outside of the AI map Node (%d)",
			entity->name_replace(), VPUSH(entity->o_Position), vertex.tNodeID); 
		F_entity_Destroy	(entity);
		return;
	}

	{
		graph_type::const_vertex_iterator	I = graph().vertices().begin();
		graph_type::const_vertex_iterator	E = graph().vertices().end();
		for ( ; I != E; ++I) 
		{
			if ((*I).second->data().tNodeID == vertex.tNodeID) {
				Msg			("! removing graph point [%s][%f][%f][%f] because it has the same AI node as another graph point",
					entity->name_replace(),VPUSH(entity->o_Position));
				F_entity_Destroy	(entity);
				return;
			}
		}
	}

	vertex.tNeighbourCount	= 0;
	Memory.mem_copy			(vertex.tVertexTypes,graph_point->m_tLocations,GameGraph::LOCATION_TYPE_COUNT*sizeof(GameGraph::_LOCATION_ID));
	vertex.tLevelID			= 0;
	vertex.tDeathPointCount = 0;
	vertex.dwPointOffset	= 0;
	
 	graph().add_vertex		(vertex,graph().vertices().size());
	F_entity_Destroy		(entity);
}

void CGameGraphBuilder::load_graph_points	(const float &start, const float &amount)
{
	Msg						("Loading graph points");
  	
	string_path				spawn_file_name;
	strconcat				(sizeof(spawn_file_name), spawn_file_name, *m_level_name, "level.spawn");

 	IReader					*reader = FS.r_open(spawn_file_name);
	u32						id;
	NET_Packet				net_packet;
	for	(
			IReader *chunk = reader->open_chunk_iterator(id);
			chunk;
			chunk = reader->open_chunk_iterator(id, chunk)
		)
	{
		net_packet.B.count	= chunk->length();
		chunk->r			(net_packet.B.data,net_packet.B.count);
		load_graph_point	(net_packet);
 	}
	
	FS.r_close				(reader);
	Msg						("%d graph points loaded", graph().vertices().size());
}

// Cross-table 

int select_distance = 0;
template <typename T>
IC	bool sort_predicate_less(const T &first, const T &second)
{
	return					(first.first < second.first);
}

template <typename T>
IC	bool sort_predicate_greater(const T &first, const T &second)
{
	return					(first.first > second.first);
}


void CGameGraphBuilder::mark_vertices		(u32 level_vertex_id)
{
	CLevelGraph::const_iterator			I, E;
	m_mark_stack.reserve				(8192);
	m_mark_stack.push_back				(level_vertex_id);

	for ( ; !m_mark_stack.empty(); ) {
		level_vertex_id					= m_mark_stack.back();
		m_mark_stack.resize				(m_mark_stack.size() - 1);
		CLevelGraph::CVertex			*node = level_graph().vertex(level_vertex_id);
		level_graph().begin				(level_vertex_id,I,E);
		m_marks[level_vertex_id]		= true;
		for ( ; I != E; ++I) {
			u32							next_level_vertex_id = node->link(I);
			if (!level_graph().valid_vertex_id(next_level_vertex_id))
				continue;
			
			if (!m_marks[next_level_vertex_id])
				m_mark_stack.push_back	(next_level_vertex_id);
		}
	}
}

void CGameGraphBuilder::fill_marks			(const float &start, const float &amount)
{
	m_marks.assign						(level_graph().header().vertex_count(),false);
	graph_type::const_vertex_iterator	I = graph().vertices().begin();
	graph_type::const_vertex_iterator	E = graph().vertices().end();
	for ( ; I != E; ++I)
		mark_vertices					((*I).second->data().level_vertex_id());
	m_marks.flip						();
}



void CGameGraphBuilder::fill_distances		(const float &start, const float &amount)
{
	select_distance = graph().vertices().size();
	m_distances.resize					(select_distance);
	for (auto& Dvec : m_distances)
	{
		Dvec.resize					(level_graph().header().vertex_count());			
  		for (auto& D : Dvec)
			D = u32(-1);
	}
}

void CGameGraphBuilder::fill_distances_part(const float start, const float end)
{
	m_distances.clear();
 	m_distances.resize(start);
	for (auto& Dvec : m_distances)
	{
 		Dvec.resize(level_graph().header().vertex_count());
   		for (auto& D : Dvec)
			D = u32(-1);
	}
}

void CGameGraphBuilder::recursive_update	(const u32 &game_vertex_id, const float &start, const float &amount)
{
	u32							level_vertex_id = graph().vertex(game_vertex_id)->data().level_vertex_id();
	xr_vector<u32>				&distances = m_distances[game_vertex_id];
	m_distances[m_results[level_vertex_id]][level_vertex_id]	= u32(-1);

	m_current_fringe.reserve	(distances.size());
	m_next_fringe.reserve		(distances.size());
	distances.assign			(distances.size(),u32(-1));
	m_current_fringe.push_back	(level_vertex_id);

	u32							curr_dist = 0;
	u32							total_count = 0;
	u32							vertex_count = graph().header().vertex_count();

	float						amount_i = amount/(	float(vertex_count)*float(level_graph().header().vertex_count()) );

	for ( ; !m_current_fringe.empty(); ) {
		xr_vector<u32>::iterator			I = m_current_fringe.begin();
		xr_vector<u32>::iterator			E = m_current_fringe.end();
		for ( ; I != E; ++I) {
			u32								*result = &m_results[*I];
			VERIFY							(curr_dist < m_distances[*result][*I]);
			*result							= game_vertex_id;

			distances[*I]					= curr_dist;
			CLevelGraph::const_iterator		i, e;
			CLevelGraph::CVertex			*node = level_graph().vertex(*I);
			level_graph().begin				(*I,i,e);
			for ( ; i != e; ++i) {
				u32							dwNexNodeID = node->link(i);
				if (!level_graph().valid_vertex_id(dwNexNodeID))											continue;
 				if (m_marks[dwNexNodeID])																	continue;
 				if (distances[dwNexNodeID] <= curr_dist)													continue;
 				if (m_distances[m_results[dwNexNodeID]][dwNexNodeID] <= (curr_dist + 1))					continue;

				m_next_fringe.push_back		(dwNexNodeID);
				m_marks[dwNexNodeID]		= true;
			}
		}
		
		I = m_current_fringe.begin();
		E = m_current_fringe.end();
		for ( ; I != E; ++I)
			m_marks[*I]		= false;

		total_count			+= m_current_fringe.size();
		m_current_fringe	= m_next_fringe;
		m_next_fringe.clear	();
		++curr_dist;
	}
}

void CGameGraphBuilder::iterate_distances(const float& start, const float& amount)
{  
	m_results.assign					(level_graph().header().vertex_count(), 0);			  
	
	float								amount_i = amount/float(select_distance); 

	for (int i=0, n=(int)select_distance; i<n; ++i)  
	{
		if (i) 
		{
			for (int k=0, kn=(int)level_graph().header().vertex_count(); k<kn; ++k) 
				m_distances[i][k]		= m_distances[i - 1][k];
		}

		recursive_update				(i,start + amount_i*float(i),amount_i);
	}
}

void CGameGraphBuilder::save_cross_table	(const float &start, const float &amount)
{
	Msg									("Saving cross table");
	CMemoryWriter						tMemoryStream;
	CGameLevelCrossTable::CHeader		tCrossTableHeader;
	
	tCrossTableHeader.dwVersion			= XRAI_CURRENT_VERSION;
	tCrossTableHeader.dwNodeCount		= level_graph().header().vertex_count();
	tCrossTableHeader.dwGraphPointCount = graph().header().vertex_count();
	tCrossTableHeader.m_level_guid		= level_graph().header().guid();
	tCrossTableHeader.m_game_guid		= m_graph_guid;
	
	tMemoryStream.open_chunk			(CROSS_TABLE_CHUNK_VERSION);
	tMemoryStream.w						(&tCrossTableHeader,sizeof(tCrossTableHeader));
	tMemoryStream.close_chunk			();
	
	tMemoryStream.open_chunk			(CROSS_TABLE_CHUNK_DATA);

	for (int i=0, n = level_graph().header().vertex_count(); i<n; i++) 
	{
		CGameLevelCrossTable::CCell	tCrossTableCell;
		tCrossTableCell.tGraphIndex = (GameGraph::_GRAPH_ID)m_results[i];
		VERIFY						(graph().header().vertex_count() > tCrossTableCell.tGraphIndex);
		tCrossTableCell.fDistance	= float(m_distances[tCrossTableCell.tGraphIndex][i])*level_graph().header().cell_size();
		tMemoryStream.w				(&tCrossTableCell,sizeof(tCrossTableCell));
	}
 
	tMemoryStream.close_chunk();
	tMemoryStream.save_to	(m_cross_table_name);

	m_marks.clear			();
	m_mark_stack.clear		();
	m_distances.clear		();
	m_current_fringe.clear	();
	m_next_fringe.clear		();
}

void CGameGraphBuilder::build_cross_table	(const float &start, const float &amount)
{
	Msg						("Building cross table");
	fill_marks				(start + 0.000000f*amount,0.018725f*amount);
	fill_distances			(start + 0.018725f*amount,0.183732f*amount);
	iterate_distances		(start + 0.202457f*amount,0.757202f*amount);
	save_cross_table		(start + 0.959659f*amount,0.040327f*amount);
	load_cross_table		(start + 0.999986f*amount,0.000014f*amount);
}

// build - graph

void CGameGraphBuilder::load_cross_table	(const float &start, const float &amount)
{
	m_cross_table			= xr_new<CGameLevelCrossTable>(m_cross_table_name);
}

void CGameGraphBuilder::fill_neighbours		(const u32 &game_vertex_id)
{
	m_marks.assign						(level_graph().header().vertex_count(),false);
	m_current_fringe.clear				();

	u32									level_vertex_id = graph().vertex(game_vertex_id)->data().level_vertex_id();

	CLevelGraph::const_iterator			I, E;
	m_mark_stack.reserve				(8192);
	m_mark_stack.push_back				(level_vertex_id);

	for ( ; !m_mark_stack.empty(); )
	{
		level_vertex_id					= m_mark_stack.back();
		m_mark_stack.resize				(m_mark_stack.size() - 1);
		CLevelGraph::CVertex			*node = level_graph().vertex(level_vertex_id);
		level_graph().begin				(level_vertex_id,I,E);
		m_marks[level_vertex_id]		= true;
		for ( ; I != E; ++I)
		{
			u32							next_level_vertex_id = node->link(I);
			if (!level_graph().valid_vertex_id(next_level_vertex_id))
				continue;
			
			if (m_marks[next_level_vertex_id])
				continue;

			GameGraph::_GRAPH_ID		next_game_vertex_id = cross().vertex(next_level_vertex_id).game_vertex_id();
			VERIFY						(next_game_vertex_id < graph().vertices().size());
			if (next_game_vertex_id != (GameGraph::_GRAPH_ID)game_vertex_id) {
				if	(
						std::find(
							m_current_fringe.begin(),
							m_current_fringe.end(),
							next_game_vertex_id
						)
						==
						m_current_fringe.end()
					)
					m_current_fringe.push_back	(next_game_vertex_id);
				continue;
			}

			m_mark_stack.push_back		(next_level_vertex_id);
		}
	}
}

float CGameGraphBuilder::path_distance		(const u32 &game_vertex_id0, const u32 &game_vertex_id1)
{
	R_ASSERT					(m_graph_engine);

	graph_type::CVertex		&vertex0 = *graph().vertex(game_vertex_id0);
	graph_type::CVertex		&vertex1 = *graph().vertex(game_vertex_id1);

	typedef GraphEngineSpace::CStraightLineParams	CStraightLineParams;
	CStraightLineParams		parameters(vertex0.data().level_point(),vertex1.data().level_point());

	float					pure_distance = vertex0.data().level_point().distance_to_xz(vertex1.data().level_point());
 	VERIFY					(pure_distance < parameters.max_range);

	u32						level_vertex_id = level_graph().check_position_in_direction(vertex0.data().level_vertex_id(),vertex0.data().level_point(),vertex1.data().level_point());
	if (level_graph().valid_vertex_id(level_vertex_id))
		return				(pure_distance);

	bool					successfull = 
		m_graph_engine->search(
			level_graph(),
			vertex0.data().level_vertex_id(),
			vertex1.data().level_vertex_id(),
			&m_path,
			parameters
		);

	if (successfull)
		return				(parameters.m_distance);

	Msg						("Cannot build path from [%d] to [%d]",game_vertex_id0,game_vertex_id1);
	Msg						("Cannot build path from [%f][%f][%f] to [%f][%f][%f]",VPUSH(vertex0.data().level_point()),VPUSH(vertex1.data().level_point()));
	R_ASSERT2				(false,"Cannot build path, check AI map");

	return					(flt_max);
}

void CGameGraphBuilder::generate_edges		(const u32 &game_vertex_id)
{
	graph_type::CVertex		*vertex = graph().vertex(game_vertex_id);

	xr_vector<u32>::const_iterator	I = m_current_fringe.begin();
	xr_vector<u32>::const_iterator	E = m_current_fringe.end();
	for ( ; I != E; ++I) {
		VERIFY				(!vertex->edge(*I));
		float				distance = path_distance(game_vertex_id,*I);
		graph().add_edge	(game_vertex_id,*I,distance);
	}
}

void CGameGraphBuilder::generate_edges		(const float &start, const float &amount)
{
	graph_type::const_vertex_iterator	I = graph().vertices().begin();
	graph_type::const_vertex_iterator	E = graph().vertices().end();
	for ( ; I != E; ++I) {
		fill_neighbours		((*I).second->vertex_id());
		generate_edges		((*I).second->vertex_id());
	}
}

void CGameGraphBuilder::create_tripples		(const float &start, const float &amount)
{
	graph_type::const_vertex_iterator	I = graph().vertices().begin();
	graph_type::const_vertex_iterator	E = graph().vertices().end();
	for ( ; I != E; ++I) {
		graph_type::const_iterator	i = (*I).second->edges().begin();
		graph_type::const_iterator	e = (*I).second->edges().end();
		for ( ; i != e; ++i) {
			if (((*i).vertex_id() < (*I).first) && graph().edge((*i).vertex_id(),(*I).first))
				continue;

			const graph_type::CEdge	*edge = graph().vertex((*i).vertex_id())->edge((*I).first);

			m_tripples.push_back	(
				std::make_pair(
					_min(
						(*i).weight(),
						edge ? edge->weight() : (*i).weight()
					),
					std::make_pair(
						(*I).first,
						(*i).vertex_id()
					)
				)
			);
		}
	}

	std::sort				(m_tripples.begin(),m_tripples.end(),sort_predicate_greater<TRIPPLE>);
}

void CGameGraphBuilder::process_tripple		(const TRIPPLE &tripple)
{
	const graph_type::CVertex	&vertex0 = *graph().vertex(tripple.second.first);
	const graph_type::CVertex	&vertex1 = *graph().vertex(tripple.second.second);

	graph_type::const_iterator	I = vertex0.edges().begin();
	graph_type::const_iterator	E = vertex0.edges().end();
	for ( ; I != E; ++I) {
		if ((*I).vertex_id() == tripple.second.second)
			continue;

		const graph_type::CEdge	*edge;

		edge					= vertex1.edge((*I).vertex_id());
		if (edge) {
			VERIFY				(_min((*I).weight(),graph().edge((*I).vertex_id(),tripple.second.first) ? graph().edge((*I).vertex_id(),tripple.second.first)->weight() : (*I).weight()) <= tripple.first);
			VERIFY				(_min(edge->weight(),graph().edge(edge->vertex_id(),tripple.second.second) ? graph().edge(edge->vertex_id(),tripple.second.second)->weight() : (*I).weight()) <= tripple.first);
			if (vertex0.edge(tripple.second.second))
				graph().remove_edge	(tripple.second.first,tripple.second.second);
			if (vertex1.edge(tripple.second.first))
				graph().remove_edge	(tripple.second.second,tripple.second.first);
			return;
		}

		edge					= graph().vertex((*I).vertex_id())->edge(tripple.second.second);
		if (edge) {
			VERIFY				(_min((*I).weight(),graph().edge((*I).vertex_id(),tripple.second.first) ? graph().edge((*I).vertex_id(),tripple.second.first)->weight() : (*I).weight()) <= tripple.first);
			VERIFY				(edge->weight() <= tripple.first);
			if (vertex0.edge(tripple.second.second))
				graph().remove_edge	(tripple.second.first,tripple.second.second);
			if (vertex1.edge(tripple.second.first))
				graph().remove_edge	(tripple.second.second,tripple.second.first);
			return;
		}
	}
	return;
}

void CGameGraphBuilder::optimize_graph		(const float &start, const float &amount)
{
	create_tripples				(start + .00f, amount*.50f);
 	for (auto& T : m_tripples)
		process_tripple			(T);
}

void CGameGraphBuilder::save_graph			(const float &start, const float &amount)
{
	Msg						("Saving graph");

	// header
	CMemoryWriter				writer;
	CGameGraph::CHeader			header;
	header.m_version			= XRAI_CURRENT_VERSION;
	VERIFY						(graph().vertices().size() < (u32(1) << (8*sizeof(GameGraph::_GRAPH_ID))));
	header.m_vertex_count		= (GameGraph::_GRAPH_ID)graph().vertices().size();
	VERIFY						(graph().edge_count() < (u32(1) << (8*sizeof(GameGraph::_GRAPH_ID))));
	header.m_edge_count			= (GameGraph::_GRAPH_ID)graph().edge_count();
	header.m_death_point_count	= 0;
	header.m_guid				= m_graph_guid;

	// levels
	CGameGraph::SLevel			level;
	level.m_offset.set			(0,0,0);
	level.m_id					= 0;
	level.m_name				= m_level_name;
	level.m_section				= "";
	level.m_guid				= level_graph().header().guid();

	header.m_levels.insert		(std::make_pair(level.m_id,level));

	header.save					(&writer);

	{
		u32								edge_offset = graph().vertices().size()*sizeof(CGameGraph::CVertex);

		graph_type::const_vertex_iterator	I = graph().vertices().begin();
		graph_type::const_vertex_iterator	E = graph().vertices().end();
		for ( ; I != E; ++I) {
			CGameGraph::CVertex		&vertex = (*I).second->data();

			VERIFY					((*I).second->edges().size() < 256);
			vertex.tNeighbourCount	= (u8)(*I).second->edges().size();
			vertex.dwEdgeOffset		= edge_offset;
			edge_offset				+= vertex.tNeighbourCount*sizeof(CGameGraph::CEdge);

			writer.w				(&vertex,sizeof(CGameGraph::CVertex));
		}
	}
	
	{
		graph_type::const_vertex_iterator	I = graph().vertices().begin();
		graph_type::const_vertex_iterator	E = graph().vertices().end();
		for ( ; I != E; ++I) {
			graph_type::const_iterator	i = (*I).second->edges().begin();
			graph_type::const_iterator	e = (*I).second->edges().end();
			for ( ; i != e; ++i) {
				GameGraph::CEdge			edge;
				VERIFY						((*i).vertex_id() < (u32(1) << (8*sizeof(GameGraph::_GRAPH_ID))));
				edge.m_vertex_id			= (GameGraph::_GRAPH_ID)(*i).vertex_id();
				edge.m_path_distance		= (*i).weight();

				writer.w					(&edge.m_vertex_id,sizeof(edge.m_vertex_id));
				writer.w_float				(edge.m_path_distance);
			}
		}
	}

	writer.save_to				(m_graph_name);

	Msg							("%d bytes saved",int(writer.size()));
}

void CGameGraphBuilder::build_graph			(const float &start, const float &amount)
{
	Msg("[CGameGraphBuilder] Building graph: ");

	m_graph_engine			= xr_new<CGraphEngine>(level_graph().header().vertex_count());

	Fbox Volume = level_graph().header().box();
	Msg("[CGameGraphBuilder] Current Level Graph: {%f, %f, %f} -> {%f, %f, %f}", VPUSH(Volume.min), VPUSH(Volume.max) );
	Msg("[CGameGraphBuilder] Current Graph : %p | vertexs: %u size", &graph(), graph().vertices().size());


	Msg("[CGameGraphBuilder] State: Generate Edges");
  	generate_edges			(start + 0.067204f*amount, amount*0.922647f);
	xr_delete				(m_graph_engine);

	Msg("[CGameGraphBuilder] State: Optimize Graph");
	optimize_graph			(start + 0.992031f*amount, amount*0.000454f);

	Msg("[CGameGraphBuilder] State: Save Graph");
	save_graph				(start + 0.992485f*amount, amount*0.007515f);
}

void CGameGraphBuilder::build_graph			(
		LPCSTR graph_name,
		LPCSTR cross_table_name,
		LPCSTR level_name
	)
{
	Phase					("Building level game graph");
	Msg						("level \"%s\"",level_name);

	m_graph_name			= graph_name;
	m_cross_table_name		= cross_table_name;
	m_level_name			= level_name;

	// CLearing temp data
	m_tripples.clear();

	m_marks.clear();
	m_mark_stack.clear();

	m_current_fringe.clear();
	m_next_fringe.clear();

	m_distances.clear();
	m_results.clear();

	xr_delete(m_graph);
	xr_delete(m_cross_table);
	xr_delete(m_level_graph);

	// Recrating !

	create_graph			(0.000000f,0.000047f);
	load_level_graph		(0.000047f,0.002470f);
	load_graph_points		(0.002517f,0.111812f);

	build_cross_table		(0.114329f,0.773423f);
	build_graph				(0.887752f,0.112248f);

	Msg						("Level graph is generated successfully");
}
