////////////////////////////////////////////////////////////////////////////
//	Module 		: level_spawn_constructor.cpp
//	Created 	: 16.10.2004
//  Modified 	: 16.10.2004
//	Author		: Dmitriy Iassenev
//	Description : Level spawn constructor
////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "level_spawn_constructor.h"
#include "game_level_cross_table.h"
#include "level_graph.h"
#include "graph_engine.h"
#include "xrmessages.h"
#include "xrServer_Objects_ALife_All.h"
#include "factory_api.h"
#include "clsid_game.h"
#include "game_base_space.h"
#include "game_spawn_constructor.h"
#include "patrol_path_storage.h"
#include "space_restrictor_wrapper.h"
#include "object_broker.h"
#include "restriction_space.h"

#define IGNORE_ZERO_SPAWN_POSITIONS

const float y_shift_correction = .15f;

CLevelSpawnConstructor::~CLevelSpawnConstructor					()
{
	GRAPH_POINT_STORAGE::iterator	I = m_graph_points.begin();
	GRAPH_POINT_STORAGE::iterator	E = m_graph_points.end();
	for ( ; I != E; ++I)
		F_entity_Destroy			((CSE_Abstract*&)(*I));

	VERIFY					(!m_level_graph);
	VERIFY					(!m_cross_table);
	VERIFY					(!m_graph_engine);
}

IC	const CGameGraph &CLevelSpawnConstructor::game_graph		() const
{
	return					(m_game_spawn_constructor->game_graph());
}

IC	CLevelSpawnConstructor::LEVEL_CHANGER_STORAGE &CLevelSpawnConstructor::level_changers() const
{
	return					(m_game_spawn_constructor->level_changers());
}

IC	u32	CLevelSpawnConstructor::level_id						(shared_str level_name) const
{
	return					(m_game_spawn_constructor->level_id(*level_name));
}

IC	const CLevelGraph &CLevelSpawnConstructor::level_graph			() const
{
	return					(*m_level_graph);
}

IC	const CGameLevelCrossTable &CLevelSpawnConstructor::cross_table	() const
{
	return					(*m_cross_table);
}

IC	CGraphEngine &CLevelSpawnConstructor::graph_engine			() const
{
	return					(*m_graph_engine);
}

void CLevelSpawnConstructor::init								()
{
	// loading level graph
	string_path				file_name;
	FS.update_path			(file_name,"$game_levels$",*m_level.name());
	xr_strcat				(file_name,"\\");

	Msg("$[SpawnConstructor] LEVEL GRAPH: %s", file_name);
	m_level_graph			= xr_new<CLevelGraph>(file_name);
	
	// loading cross table
	m_game_spawn_constructor->game_graph().set_current_level	(game_graph().header().level(*m_level.name()).id());
	m_cross_table			= &game_graph().cross_table();

	// loading patrol paths
	FS.update_path			(file_name,"$game_levels$",*m_level.name());
	xr_strcat					(file_name,"\\level.game");
	if (FS.exist(file_name)) 
	{
		IReader				*stream	= FS.r_open(file_name);
		VERIFY				(stream);
		m_game_spawn_constructor->patrol_path_storage().load_raw(&level_graph(),&cross_table(),&game_graph(),*stream);
		FS.r_close			(stream);
	}
}

CSE_Abstract *CLevelSpawnConstructor::create_object						(IReader *chunk)
{
	NET_Packet				net_packet;
	net_packet.B.count		= chunk->length();
	chunk->r				(net_packet.B.data,net_packet.B.count);
//	we do not need to close chunk since we iterate on them
//	chunk->close			();
	u16						ID;
	net_packet.r_begin		(ID);
	R_ASSERT2				(M_SPAWN==ID,"ID doesn't match to the spawn-point ID!");
	string64				section_name;
	net_packet.r_stringZ	(section_name);
	CSE_Abstract			*abstract = F_entity_Create(section_name);
	if (!abstract) {
		string256			temp;
		xr_sprintf				(temp,"Can't create entity '%s' !\n",section_name);
		R_ASSERT2			(abstract,temp);
	}
	abstract->Spawn_Read	(net_packet);
	return					(abstract);
}

void CLevelSpawnConstructor::add_graph_point					(CSE_Abstract			*abstract)
{
	CSE_ALifeGraphPoint		*graph_point = smart_cast<CSE_ALifeGraphPoint*>(abstract);
	R_ASSERT				(graph_point);
	m_graph_points.push_back(graph_point);
}

void CLevelSpawnConstructor::add_story_object					(CSE_ALifeDynamicObject *dynamic_object)
{
	m_game_spawn_constructor->add_story_object	(dynamic_object->m_story_id,dynamic_object,*m_level.name());
}

void CLevelSpawnConstructor::add_space_restrictor				(CSE_ALifeDynamicObject *dynamic_object)
{
	CSE_ALifeSpaceRestrictor		*space_restrictor = smart_cast<CSE_ALifeSpaceRestrictor*>(dynamic_object);
	if (!space_restrictor)
		return;

	if (space_restrictor->m_space_restrictor_type == RestrictionSpace::eRestrictorTypeNone)
		return;

	if (!space_restrictor->m_flags.test(CSE_ALifeObject::flCheckForSeparator))
		return;

	m_space_restrictors.push_back	(xr_new<CSpaceRestrictorWrapper>(space_restrictor));
}

void CLevelSpawnConstructor::add_level_changer					(CSE_Abstract			*abstract)
{
	CSE_ALifeLevelChanger	*level_changer = smart_cast<CSE_ALifeLevelChanger*>(abstract);
	R_ASSERT				(level_changer);
	m_game_spawn_constructor->add_level_changer	(level_changer);
	m_level_changers.push_back	(level_changer);
}

void CLevelSpawnConstructor::add_free_object					(CSE_Abstract			*abstract)
{
	m_game_spawn_constructor->add_object		(abstract);
}

void CLevelSpawnConstructor::load_objects						()
{
	// loading spawn points
	string_path					file_name;
	FS.update_path				(file_name,"$game_levels$",*m_level.name());
	xr_strcat					(file_name,"\\level.spawn");

	Msg("! Loading Level Spawn: %s", file_name);

	IReader						*level_spawn = FS.r_open(file_name);
	u32							id;
	IReader						*chunk = level_spawn->open_chunk_iterator(id);
	for ( ; chunk; chunk = level_spawn->open_chunk_iterator(id,chunk)) {
		CSE_Abstract			*abstract = create_object(chunk);
		if (abstract->m_tClassID == CLSID_AI_GRAPH) {
			add_graph_point		(abstract);
			continue;
		}

		if (!abstract->m_gameType.MatchType(eGameIDSingle)) 
		{
			F_entity_Destroy	(abstract);
			continue;
		}

		CSE_ALifeObject			*alife_object = smart_cast<CSE_ALifeObject*>(abstract);
		if (!alife_object) {
			F_entity_Destroy	(abstract);
			continue;
		}

		CSE_ALifeCreatureActor	*actor = smart_cast<CSE_ALifeCreatureActor*>(alife_object);
		if (actor) {
			R_ASSERT3			(!m_actor,"Too many actors on the level ",*m_level.name());
			m_actor				= actor;
		}

		m_spawns.push_back		(alife_object);

		CSE_ALifeDynamicObject	*dynamic_object = smart_cast<CSE_ALifeDynamicObject*>(alife_object);
		if (dynamic_object) {
			add_story_object	(dynamic_object);
			add_space_restrictor(dynamic_object);
		}

		if (smart_cast<CSE_ALifeLevelChanger*>(abstract))
			add_level_changer	(abstract);

		add_free_object			(alife_object);
	}
	
	FS.r_close					(level_spawn);

	R_ASSERT2					(!m_spawns.empty(),"There are no spawn-points!");
}
 
void CLevelSpawnConstructor::correct_objects					()
{
	u32						m_level_graph_vertex_id = u32(-1);
	u32						dwStart = game_graph().header().vertex_count(), dwFinish = game_graph().header().vertex_count(), dwCount = 0;
	for (u32 i=0; i<game_graph().header().vertex_count(); ++i)
		if (game_graph().vertex(i)->level_id() == m_level.id()) {
			if (m_level_graph_vertex_id == u32(-1))
				m_level_graph_vertex_id = i;
			dwCount++;
		}
	
	for (int i=0; i<(int)game_graph().header().vertex_count(); i++)
		if (game_graph().vertex(i)->level_id() == m_level.id()) {
			if (dwStart > (u32)i)
				dwStart = (u32)i;
		}
		else {
			if ((dwStart <= (u32)i) && (dwFinish > (u32)i)) {
				dwFinish = i;
				break;
			}
		}
	if (dwStart >= dwFinish) {
		string4096			S;
		xr_sprintf				(S,"There are no graph vertices in the game graph for the level '%s' !\n",*m_level.name());
		R_ASSERT2			(dwStart < dwFinish,S);
	}

	for (int i=0; i<(int)m_spawns.size(); i++) 
	{
		// if (game_graph().valid_vertex_id(m_spawns[i]->m_tGraphID))
 		//	Msg("Graph Vertex not valid id[%d] name[%s] position [%f][%f][%f]", m_spawns[i]->ID, m_spawns[i]->name(), VPUSH(m_spawns[i]->position()));
 		// if (!level_graph().valid_vertex_id(m_spawns[i]->m_tGraphID))
 		// 	Msg("Node Vertex not valid id [%d] name[%s] position [%f][%f][%f]", m_spawns[i]->ID, m_spawns[i]->name(), VPUSH(m_spawns[i]->position()));
 
		if (!m_spawns[i]->used_ai_locations())
		{
			m_spawns[i]->m_tGraphID = (GameGraph::_GRAPH_ID)m_level_graph_vertex_id;
			m_spawns[i]->m_fDistance = 0.f;
			m_spawns[i]->m_tNodeID = game_graph().vertex(m_level_graph_vertex_id)->level_vertex_id();
			continue;
		}


		Fvector				position = m_spawns[i]->o_Position;
		position.y			+= y_shift_correction;
		m_spawns[i]->m_tNodeID = level_graph().vertex(u32(-1),position);
		VERIFY				(level_graph().valid_vertex_id(m_spawns[i]->m_tNodeID));
		if (m_spawns[i]->used_ai_locations() && !level_graph().inside(level_graph().vertex(m_spawns[i]->m_tNodeID),position)) {
			Fvector			new_position = level_graph().vertex_position(m_spawns[i]->m_tNodeID);
			clMsg			("[%s][%s][%s] : position changed from [%f][%f][%f] -> [%f][%f][%f]",*m_level.name(),*m_spawns[i]->s_name,m_spawns[i]->name_replace(),VPUSH(position),VPUSH(new_position));
			m_spawns[i]->o_Position	= new_position;
		}
		u32					dwBest = cross_table().vertex(m_spawns[i]->m_tNodeID).game_vertex_id();
		if (game_graph().vertex(dwBest)->level_id() != m_level.id()) {
			string4096	S1;
			char		*S = S1;
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]), "Corresponding graph vertex for the spawn point is located on the ANOTHER level\n",m_spawns[i]->name_replace());
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Current level  : [%d][%s]\n",m_level.id(),*game_graph().header().level(m_level.id()).name());
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Conflict level : [%d][%s]\n",game_graph().vertex(dwBest)->level_id(),*game_graph().header().level(game_graph().vertex(dwBest)->level_id()).name());
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Probably, you filled offsets in \"game_levels.ltx\" incorrect");
			R_ASSERT2	(game_graph().vertex(dwBest)->level_id() == m_level.id(),S1);
		}

		float				fCurrentBestDistance = cross_table().vertex(m_spawns[i]->m_tNodeID).distance();
		if (dwBest == u32(-1)) {
			string4096	S1;
			char		*S = S1;
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Can't find a corresponding GRAPH VERTEX for the spawn-point %s\n",m_spawns[i]->name_replace());
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Level ID    : %d\n",m_level.id());
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Spawn index : %d\n",i);
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Spawn node  : %d\n",m_spawns[i]->m_tNodeID);
			S			+= xr_sprintf(S,sizeof(S1) - (S1 - &S[0]),"Spawn point : [%7.2f][%7.2f][%7.2f]\n",m_spawns[i]->o_Position.x,m_spawns[i]->o_Position.y,m_spawns[i]->o_Position.z);
			R_ASSERT2	(dwBest != -1,S1);
		}
		m_spawns[i]->m_tGraphID		= (GameGraph::_GRAPH_ID)dwBest;
		m_spawns[i]->m_fDistance	= fCurrentBestDistance;
	}
}

void CLevelSpawnConstructor::correct_level_changers				()
{
	LEVEL_CHANGER_STORAGE::const_iterator	I = m_level_changers.begin();
	LEVEL_CHANGER_STORAGE::const_iterator	E = m_level_changers.end();
	for ( ; I != E; ++I) {
		Fvector				position = (*I)->o_Position;
		position.y			+= y_shift_correction;
		(*I)->m_tNodeID		= level_graph().vertex(u32(-1),position);
		VERIFY				(level_graph().valid_vertex_id((*I)->m_tNodeID));

		u32					dwBest = cross_table().vertex((*I)->m_tNodeID).game_vertex_id();
		VERIFY				(game_graph().vertex(dwBest)->level_id() == m_level.id());
		(*I)->m_tGraphID	= (GameGraph::_GRAPH_ID)dwBest;

		(*I)->m_fDistance	= cross_table().vertex((*I)->m_tNodeID).distance();
	}
}

struct remove_too_far_predicate {
	float				m_radius_sqr;
	const CLevelGraph	*m_graph;
	Fvector				m_position;

	IC			remove_too_far_predicate	(const CLevelGraph *graph, const Fvector &position, float radius)
	{
		VERIFY			(graph);
		m_graph			= graph;
		m_position		= position;
		m_radius_sqr	= _sqr(radius);
	}

	IC	bool	operator()					(const u32 &vertex_id) const
	{
		return			(m_graph->vertex_position(vertex_id).distance_to_sqr(m_position) > m_radius_sqr);
	}
};

void CLevelSpawnConstructor::fill_level_changers				()
{
	for (u32 i=0, n=(u32)level_changers().size(); i<n; ++i)
	{
		if (level_id(level_changers()[i]->m_caLevelToChange) != m_level.id())
			continue;

		// Msg("[LevelChanger] [%s]: point to [%s] )",
		// 	level_changers()[i]->name_replace(),
		// 	level_changers()[i]->m_caLevelToChange.c_str(),
		// 	level_changers()[i]->m_caLevelPointToChange.c_str());

		bool found = false;
		for (auto& GP : m_graph_points)
		{
			if (!xr_strcmp(*level_changers()[i]->m_caLevelPointToChange, GP->name_replace()))
			{
 				bool ok = false;
				for (u32 ii = 0, nn = game_graph().header().vertex_count(); ii < nn; ++ii) 
				{
					if ((game_graph().vertex(ii)->level_id() != m_level.id()) || 
						!game_graph().vertex(ii)->level_point().similar(GP->o_Position, .001f))
						continue;

					level_changers()[i]->m_tNextGraphID		= (GameGraph::_GRAPH_ID)ii;
					level_changers()[i]->m_tNextPosition	= GP->o_Position;
					level_changers()[i]->m_tAngles			= GP->o_Angle;
					level_changers()[i]->m_dwNextNodeID		= game_graph().vertex(ii)->level_vertex_id();
					ok = true;
					break;
				}

				if (!ok)
					clMsg("Cannot find[%s] a correspndance between graph and graph points from level editor!", *level_changers()[i]->m_caLevelToChange);
 				
				level_changers().erase(level_changers().begin() + i);
				--i;
				--n;
				found = true;
				break;
			}
 		}
 
		if (!found)
 			clMsg("Graph point %s not found", *level_changers()[i]->m_caLevelPointToChange, level_changers()[i]->name_replace());
	}
}

void CLevelSpawnConstructor::update_artefact_spawn_positions	()
{
	u32									level_point_count = m_game_spawn_constructor->level_point_count();
	SPAWN_STORAGE::iterator				I = m_spawns.begin();
	SPAWN_STORAGE::iterator				E = m_spawns.end();
	for ( ; I != E; ++I) {
		CSE_Abstract					*abstract = *I;
		CSE_ALifeObject					*alife_object = smart_cast<CSE_ALifeObject*>(abstract);
 		R_ASSERT2						(alife_object,"Non-ALife object!");
		VERIFY							(game_graph().vertex(alife_object->m_tGraphID)->level_id() == m_level.id());
 		CSE_ALifeAnomalousZone			*zone = smart_cast<CSE_ALifeAnomalousZone*>(abstract);
		if (zone) {
			zone->m_artefact_position_offset		= level_point_count;
			level_point_count			+= zone->m_artefact_spawn_count;
		}
	}

	m_game_spawn_constructor->add_level_points	(m_level_points);
}

void CLevelSpawnConstructor::Execute							()
{
	CTimer t; t.Start();
	load_objects						();
	init								();
	correct_objects						();
 	correct_level_changers				();
	verify_space_restrictors			();
	
	xr_delete							(m_level_graph);
	m_cross_table						= 0;
	xr_delete							(m_graph_engine);
}

void CLevelSpawnConstructor::update								()
{
	fill_level_changers					();
	update_artefact_spawn_positions		();
}

void CLevelSpawnConstructor::verify_space_restrictors			()
{
	Msg									("Level [%s] : searching for AI map separators space restrictors",*m_level.name());
	SPACE_RESTRICTORS::iterator			I = m_space_restrictors.begin();
	SPACE_RESTRICTORS::iterator			E = m_space_restrictors.end();
	for ( ; I != E; ++I) {
		VERIFY							(*I);
		
		if ((*I)->object().m_space_restrictor_type == RestrictionSpace::eRestrictorTypeNone)
			continue;

		(*I)->verify					(*m_level_graph,*m_graph_engine,m_no_separator_check);
	}

	delete_data							(m_space_restrictors);

	if (m_no_separator_check)
		Msg								("Level [%s] : no separators found",*m_level.name());
}
