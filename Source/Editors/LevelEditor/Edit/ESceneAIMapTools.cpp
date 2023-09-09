#include "stdafx.h"
#pragma hdrstop

#include "ESceneAIMapTools.h"
#include "xrLevel.h"
#include "Scene.h"
#include "ui_levelmain.h"
#include "ui_leveltools.h"
#include "ESceneAIMapControls.h"
#include "..\..\XrCore\xrPool.h"

//#define save_no_pack

#ifdef save_no_pack
// chunks
extern int AIMAP_VERSION = 0x0004;
extern int AIMAP_VERSION_2 = 0x0002;
extern int AIMAP_VERSION_3 = 0x0003;
#else 
// chunks
extern int AIMAP_VERSION = 0x0003;
extern int AIMAP_VERSION_2 = 0x0002;
#endif



extern int ai_version = -1;
//----------------------------------------------------
#define AIMAP_CHUNK_VERSION			0x0001       
#define AIMAP_CHUNK_FLAGS			0x0002
#define AIMAP_CHUNK_BOX				0x0003
#define AIMAP_CHUNK_PARAMS			0x0004
#define AIMAP_CHUNK_NODES			0x0006
#define AIMAP_CHUNK_SNAP_OBJECTS	0x0007
#define AIMAP_CHUNK_INTERNAL_DATA	0x0008
#define AIMAP_CHUNK_INTERNAL_DATA2	0x0009
//----------------------------------------------------


 

poolSS<SAINode,1024> g_ainode_pool;

void* SAINode::operator new(std::size_t size)
{
	return g_ainode_pool.create();
}

void* SAINode::operator new(std::size_t size, SAINode* src)
{
    return 			src;
}

void SAINode::operator delete(void* ptr)
{
    auto node = (SAINode*)ptr;
	g_ainode_pool.destroy(node);
}

void SAINode::PointLF(Fvector& D, float patch_size)
{
	Fvector	d;	d.set(0,-1,0);
	Fvector	v	= Pos;	
	float	s	= patch_size/2;
	v.x			-= s;
	v.z			+= s;
	Plane.intersectRayPoint(v,d,D);
}

void SAINode::PointFR(Fvector& D, float patch_size)
{
	Fvector	d;	d.set(0,-1,0);
	Fvector	v	= Pos;	
	float	s	= patch_size/2;
	v.x			+= s;
	v.z			+= s;
	Plane.intersectRayPoint(v,d,D);
}

void SAINode::PointRB(Fvector& D, float patch_size)
{
	Fvector	d;	d.set(0,-1,0);
	Fvector	v	= Pos;	
	float	s	= patch_size/2;
	v.x			+= s;
	v.z			-= s;
	Plane.intersectRayPoint(v,d,D);
}

void SAINode::PointBL(Fvector& D, float patch_size)
{
	Fvector	d;	d.set(0,-1,0);
	Fvector	v	= Pos;	
	float	s	= patch_size/2;
	v.x			-= s;
	v.z			-= s;
	Plane.intersectRayPoint(v,d,D);
}
 
void SAINode::LoadStream(IReader& F, ESceneAIMapTool* tools)
{
    if (ai_version == AIMAP_VERSION_2)
    {
        u32 			id;
        u16 			pl;
        NodePosition 	np;
        F.r(&id, 3); 			n1 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 3); 			n2 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 3); 			n3 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 3); 			n4 = (SAINode*)tools->UnpackLink(id);
        pl = F.r_u16(); 		pvDecompress(Plane.n, pl);
        F.r(&np, sizeof(np)); 	tools->UnpackPosition(Pos, np, tools->m_AIBBox, tools->m_Params);
        Plane.build(Pos, Plane.n);
        flags.assign(F.r_u8());
    }
    else
    {
        u32 			id;
        F.r(&id, 4);
        n1 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n2 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n3 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n4 = (SAINode*)tools->UnpackLink(id);

        pvDecompress(Plane.n, F.r_u16());
        F.r_fvector3(Pos);
        Plane.build(Pos, Plane.n);
        flags.assign(F.r_u8());
    }


    /*
    {
        u32 			id;
        NodePosition 	np;

        F.r(&id, 4);
        n1 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n2 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n3 = (SAINode*)tools->UnpackLink(id);
        F.r(&id, 4);
        n4 = (SAINode*)tools->UnpackLink(id);

        pvDecompress(Plane.n, F.r_u16());
        F.r(&np, sizeof(np)); 
        tools->UnpackPosition(Pos, np, tools->m_AIBBox, tools->m_Params);
        Plane.build(Pos, Plane.n);
        flags.assign(F.r_u8());
    }
    */
    
  /*  u32 max_32 = 0x00ffffff;
    
    if (n1->idx > max_32 && n2->idx > max_32 && n3->idx > max_32 && n4->idx > max_32)
    {
        Msg("n1[%d], n2[%d], n3[%d], n4[%d], pos[%f][%f][%f]", n1->idx, n2->idx, n3->idx, n4->idx, VPUSH(Pos));
    }
 */   
}

void SAINode::SaveStream(IWriter& F, ESceneAIMapTool* tools)
{
   
	u32 			id;
  
    id = n1?(u32)n1->idx:InvalidNode; F.w(&id,4);
    id = n2?(u32)n2->idx:InvalidNode; F.w(&id,4);
    id = n3?(u32)n3->idx:InvalidNode; F.w(&id,4);
    id = n4?(u32)n4->idx:InvalidNode; F.w(&id,4);
   
    F.w_u16(pvCompress(Plane.n));
    F.w_fvector3(Pos);
    F.w_u8			(flags.get());
 
    /*
    u32 			id;
    u16 			pl;
    NodePosition 	np;

    id = n1 ? (u32)n1->idx : InvalidNode_32bit; F.w(&id, 4);
    id = n2 ? (u32)n2->idx : InvalidNode_32bit; F.w(&id, 4);
    id = n3 ? (u32)n3->idx : InvalidNode_32bit; F.w(&id, 4);
    id = n4 ? (u32)n4->idx : InvalidNode_32bit; F.w(&id, 4);

    pl = pvCompress(Plane.n);	 F.w_u16(pl);
    tools->PackPosition(np, Pos, tools->m_AIBBox, tools->m_Params);
    F.w(&np, sizeof(np));
    F.w_u8(flags.get());
    */
}

ESceneAIMapTool::ESceneAIMapTool():ESceneToolBase(OBJCLASS_AIMAP)
{
    m_Shader	= 0;
    m_Flags.zero();

    m_AIBBox.invalidate	();
//    m_Header.size_y				= m_Header.aabb.max.y-m_Header.aabb.min.y+EPS_L;
	hash_Initialize();
    m_VisRadius		= 30.f;
    m_SmoothHeight	= 0.5f;
    m_BrushSize	= 1;
    m_CFModel	= 0;
}
//----------------------------------------------------

ESceneAIMapTool::~ESceneAIMapTool()
{
}
//----------------------------------------------------


void ESceneAIMapTool::Clear(bool bOnlyNodes)
{
	inherited::Clear	();
	hash_Clear			();
	for (SAINode*node:m_Nodes)
    	xr_delete		(node);
	m_Nodes.clear_and_free();
	if (!bOnlyNodes)
    {
	    //m_SnapObjects.clear	();
        m_AIBBox.invalidate	();
        ExecCommand		(COMMAND_REFRESH_SNAP_OBJECTS);
		g_ainode_pool.clear	();
        RealUpdateSnapList();
    }
}
//----------------------------------------------------

u32 ESceneAIMapTool::UnpackLink(u32& L)
{
    if (ai_version == AIMAP_VERSION_2)
    {
        u32 link = L & 0x00ffffff;
        return link;
    }

    return L;
}

void ESceneAIMapTool::CalculateNodesBBox(Fbox& bb)
{
    bb.invalidate();
	for (AINodeIt b_it=m_Nodes.begin(); b_it!=m_Nodes.end(); ++b_it)
    {
    	VERIFY(_valid((*b_it)->Pos));
    	bb.modify((*b_it)->Pos);
    }
}
//----------------------------------------------------
extern BOOL ai_map_shown;
void ESceneAIMapTool::OnActivate()
{
	inherited::OnActivate		();
    ai_map_shown				= TRUE;
}

void ESceneAIMapTool::OnFrame()
{
	if (m_Flags.is(flUpdateHL)){
    	m_Flags.set(flUpdateHL,FALSE);
        for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
			(*it)->flags.set(SAINode::flHLSelected,FALSE);
        for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++){
            SAINode& N = **it;
            if (N.flags.is(SAINode::flSelected))
                for (int k=0; k<4; k++)
                    if (N.n[k]) N.n[k]->flags.set(SAINode::flHLSelected,TRUE);
        }
    }
	if (m_Flags.is(flUpdateSnapList)) RealUpdateSnapList();
}
//----------------------------------------------------

void ESceneAIMapTool::EnumerateNodes()
{
    u32 idx			= 0;
	for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++,idx++)
    	(*it)->idx	= idx;
}

void ESceneAIMapTool::DenumerateNodes()
{
	u32 cnt=m_Nodes.size();

    if (ai_version == AIMAP_VERSION_2)
        for (AINodeIt it = m_Nodes.begin(); it != m_Nodes.end(); it++)
        {
            if (!((((u32)(*it)->n1 < cnt) || ((u32)(*it)->n1 == InvalidNode_32bit)) &&
                 (((u32)(*it)->n2 < cnt) || ((u32)(*it)->n2 == InvalidNode_32bit)) &&
                 (((u32)(*it)->n3 < cnt) || ((u32)(*it)->n3 == InvalidNode_32bit)) &&
                 (((u32)(*it)->n4 < cnt) || ((u32)(*it)->n4 == InvalidNode_32bit))
                ))
            {
                ELog.Msg(mtError, "Node: has wrong link [%3.2f, %3.2f, %3.2f], {%d,%d,%d,%d}", VPUSH((*it)->Pos), (*it)->n1, (*it)->n2, (*it)->n3, (*it)->n4);
                (*it)->n1 = 0;
                (*it)->n2 = 0;
                (*it)->n3 = 0;
                (*it)->n4 = 0;
                continue;
            }

            (*it)->n1 = ((u32)(*it)->n1 == InvalidNode_32bit) ? 0 : m_Nodes[(u32)(*it)->n1];
            (*it)->n2 = ((u32)(*it)->n2 == InvalidNode_32bit) ? 0 : m_Nodes[(u32)(*it)->n2];
            (*it)->n3 = ((u32)(*it)->n3 == InvalidNode_32bit) ? 0 : m_Nodes[(u32)(*it)->n3];
            (*it)->n4 = ((u32)(*it)->n4 == InvalidNode_32bit) ? 0 : m_Nodes[(u32)(*it)->n4];

        }
    else
        for (AINodeIt it = m_Nodes.begin(); it != m_Nodes.end(); it++)
        {
            if (!((((u32)(*it)->n1 < cnt) || ((u32)(*it)->n1 == InvalidNode)) &&
                (((u32)(*it)->n2 < cnt) || ((u32)(*it)->n2 == InvalidNode)) &&
                (((u32)(*it)->n3 < cnt) || ((u32)(*it)->n3 == InvalidNode)) &&
                (((u32)(*it)->n4 < cnt) || ((u32)(*it)->n4 == InvalidNode))
                ))
            {
                ELog.Msg(mtError, "Node: has wrong link [%3.2f, %3.2f, %3.2f], {%d,%d,%d,%d}", VPUSH((*it)->Pos), (*it)->n1, (*it)->n2, (*it)->n3, (*it)->n4);
                (*it)->n1 = 0;
                (*it)->n2 = 0;
                (*it)->n3 = 0;
                (*it)->n4 = 0;
                continue;
            }

            (*it)->n1 = ((u32)(*it)->n1 == InvalidNode) ? 0 : m_Nodes[(u32)(*it)->n1];
            (*it)->n2 = ((u32)(*it)->n2 == InvalidNode) ? 0 : m_Nodes[(u32)(*it)->n2];
            (*it)->n3 = ((u32)(*it)->n3 == InvalidNode) ? 0 : m_Nodes[(u32)(*it)->n3];
            (*it)->n4 = ((u32)(*it)->n4 == InvalidNode) ? 0 : m_Nodes[(u32)(*it)->n4];

        }

}


void check_lick(u32 link)
{
    if (link > 1024 * 1024 * 4)
    {
        Msg("Link: %d", link);
    }
}

bool ESceneAIMapTool::LoadStream(IReader& F)
{
	inherited::LoadStream	(F);

	u16 version = 0;

    R_ASSERT(F.r_chunk(AIMAP_CHUNK_VERSION,&version));
    if( version!=AIMAP_VERSION )
    {
       // ELog.DlgMsg( mtError, "AIMap: Unsupported version.");
       // return false;
    }

    ai_version = version;

    R_ASSERT(F.find_chunk(AIMAP_CHUNK_FLAGS));
    F.r				(&m_Flags,sizeof(m_Flags));

    R_ASSERT(F.find_chunk(AIMAP_CHUNK_BOX));
    F.r				(&m_AIBBox,sizeof(m_AIBBox));

    R_ASSERT(F.find_chunk(AIMAP_CHUNK_PARAMS));
    F.r				(&m_Params,sizeof(m_Params));

    R_ASSERT(F.find_chunk(AIMAP_CHUNK_NODES));
    //m_Nodes.resize	(F.r_u32());
    int ids = 0;

    AINodeVec vec;
    u32 size = F.r_u32();
    vec.resize(size);

    for (AINodeIt it = vec.begin(); it != vec.end(); it++, ids++)
    {
    	*it			= xr_new<SAINode>();
    	(*it)->LoadStream	(F,this);
    }

#ifdef _USE_NODE_POSITION_11
    u32 ch_node = version == AIMAP_VERSION ? InvalidNode_64bit : InvalidNode_32bit;
#else 
    u32 ch_node = InvalidNode_32bit;
#endif

    ids = 0;
    for (auto it = vec.begin(); it != vec.end(); it++, ids++)
    {
       // check_lick((u32)(*it)->n1);
       // check_lick((u32)(*it)->n2);
       // check_lick((u32)(*it)->n3);
       // check_lick((u32)(*it)->n4);

        (*it)->n1 = ((u32)(*it)->n1 >= ch_node) ? 0 : vec[(u32)(*it)->n1];
        (*it)->n2 = ((u32)(*it)->n2 >= ch_node) ? 0 : vec[(u32)(*it)->n2];
        (*it)->n3 = ((u32)(*it)->n3 >= ch_node) ? 0 : vec[(u32)(*it)->n3];
        (*it)->n4 = ((u32)(*it)->n4 >= ch_node) ? 0 : vec[(u32)(*it)->n4];
    }   

    for (auto node : vec)
        m_Nodes.push_back(node);


	//DenumerateNodes	();

    if (F.find_chunk(AIMAP_CHUNK_INTERNAL_DATA))
    {
    	m_VisRadius	= F.r_float();
    	m_BrushSize	= F.r_u32();
    }

    if (F.find_chunk(AIMAP_CHUNK_INTERNAL_DATA2))
    {
    	m_SmoothHeight	= F.r_float();
    }

	// snap objects
    if (F.find_chunk(AIMAP_CHUNK_SNAP_OBJECTS))
    {
    	shared_str 	buf;
		int cnt 	= F.r_u32();
        if (cnt)
        {
	        for (int i=0; i<cnt; i++)
            {
    	    	F.r_stringZ	(buf);
        	    CCustomObject* O = Scene->FindObjectByName(buf.c_str(),OBJCLASS_SCENEOBJECT);
            	if (!O)		ELog.Msg(mtError,"AIMap: Can't find snap object '%s'.",buf.c_str());
	            else		m_SnapObjects.push_back(O);
    	    }
        }
    }

    hash_FillFromNodes		();

    return true;
}
//----------------------------------------------------

bool ESceneAIMapTool::LoadStreamOFFSET(IReader& F, Fvector offset, bool ignore)
{ 
      
    if (F.find_chunk(3))
    {
        Fbox ai_box;
        F.r(&ai_box, sizeof(ai_box));
        
       // m_AIBBox = ai_box;

        u32 size = F.r_u32(); 
        SPBItem* pb = UI->ProgressStart(size, "Loading nodes...");
        
         Scene->lock();


        int id = 0;

        AINodeVec vec;
        vec.resize(size);

        //for (int i = 0; i < size; i++)
        for (auto it = vec.begin(); it != vec.end(); it++, id++)
        {   
           Fvector3 pos;
           F.r_fvector3(pos);
           Fvector3 norm;
           F.r_fvector3(norm);
           
           u8 flag = F.r_u8();

           pos.add(offset);

           u32 n1, n2, n3, n4;
           n1 = F.r_u32();
           n2 = F.r_u32();
           n3 = F.r_u32();
           n4 = F.r_u32();
  
           //SAINode* node = xr_new<SAINode>();
           //node->Plane.n = norm;
           //node->Pos = pos;
           //m_Nodes.push_back(node);

           *it = xr_new<SAINode>();
           (*it)->Plane.n = norm;
           (*it)->Pos = pos;
           
           (*it)->Plane.build(pos, norm);
           //(*it)->flags.flags = flag;
           //(*it)->idx = id;
           
           (*it)->n1 = (SAINode*) n1;
           (*it)->n2 = (SAINode*) n2;
           (*it)->n3 = (SAINode*) n3;
           (*it)->n4 = (SAINode*) n4;

           // BuildNode(pos, pos, true, true);
           // AddNode(pos, true, true, 1);

           if (id % 25048 == 0)
           {
               pb->Update(id);
               Msg("Load %d", id);
           }
        }

        //DenumerateNodes();

        for (auto it = vec.begin() ; it != vec.end();it++ )
        {
            (*it)->n1 = ((u32)(*it)->n1 >= size) ? 0 : vec[(u32)(*it)->n1];
            (*it)->n2 = ((u32)(*it)->n2 >= size) ? 0 : vec[(u32)(*it)->n2];
            (*it)->n3 = ((u32)(*it)->n3 >= size) ? 0 : vec[(u32)(*it)->n3];
            (*it)->n4 = ((u32)(*it)->n4 >= size) ? 0 : vec[(u32)(*it)->n4];
        }

        for (auto node : vec)
        {
            m_Nodes.push_back(node);
        }

        Scene->unlock();
         
        UI->ProgressEnd(pb);
       

        Msg("Box AI min[%f][%f][%f], max[%f][%f][%f]", VPUSH(ai_box.min), VPUSH(ai_box.max));

    
        hash_FillFromNodes();

        

        return true;
    }
     

    if (F.find_chunk(2))
    {
        Msg("!!! OLD VERSION AI EXPORT V2");
      //  return false;

        u32 size = F.r_u32();

        AINodeVec vec;
        vec.resize(size);
        SPBItem* pb = UI->ProgressStart(size, "Loading nodes...");

        int id = 0;

        for (auto it = vec.begin(); it != vec.end(); it++, id++)
        {
            Fvector3 pos;
            F.r_fvector3(pos);
            pos.add(offset);

            AddNode(pos, ignore, true, 1);

            if (id % 512 == 0)
            {
                pb->Update(id);
            }
        }

        UI->ProgressEnd(pb);
    }

}

#define ver2

void ESceneAIMapTool::SaveStreamPOS(IWriter& write)
{
#ifdef ver2
    write.open_chunk(2);

    write.w_u32(m_Nodes.size());
    for (auto node : m_Nodes)
        write.w_fvector3(node->Pos);
    write.close_chunk();
#else 

    write.open_chunk(3);
   
    Fbox ai_box = m_AIBBox;

    write.w(&ai_box,sizeof(ai_box));
    write.w_u32(m_Nodes.size());
    
    EnumerateNodes();

    for (auto node : m_Nodes)
    {
       write.w_fvector3(node->Pos);
       write.w_fvector3(node->Plane.n);   
       write.w_u8(node->flags.get());

       write.w_u32(!node->n1 ? InvalidNode : node->n1->idx );
       write.w_u32(!node->n2 ? InvalidNode : node->n2->idx);
       write.w_u32(!node->n3 ? InvalidNode : node->n3->idx);
       write.w_u32(!node->n4 ? InvalidNode : node->n4->idx);
    }

    write.close_chunk();
#endif
  //
}

void ESceneAIMapTool::SelectNode(u32 id)
{
    m_Nodes[id]->flags.set(SAINode::flSelected, TRUE);
}

bool ESceneAIMapTool::LoadSelection(IReader& F)
{
	Clear();
	return LoadStream(F);
}

void ESceneAIMapTool::OnSynchronize()
{
	RealUpdateSnapList	();
}
//----------------------------------------------------

void ESceneAIMapTool::SaveStream(IWriter& F)
{
	inherited::SaveStream	(F);

	F.open_chunk	(AIMAP_CHUNK_VERSION);
	F.w_u16			(AIMAP_VERSION);
	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_FLAGS);
    F.w				(&m_Flags,sizeof(m_Flags));
	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_BOX);
    F.w				(&m_AIBBox,sizeof(m_AIBBox));
	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_PARAMS);
    F.w				(&m_Params,sizeof(m_Params));
	F.close_chunk	();

    EnumerateNodes	();
	F.open_chunk	(AIMAP_CHUNK_NODES);
    F.w_u32			(m_Nodes.size());
	for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
    	(*it)->SaveStream	(F,this);

	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_INTERNAL_DATA);
    F.w_float		(m_VisRadius);
    F.w_u32			(m_BrushSize);
	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_INTERNAL_DATA2);
    F.w_float		(m_SmoothHeight);
	F.close_chunk	();

	F.open_chunk	(AIMAP_CHUNK_SNAP_OBJECTS);
    F.w_u32			(m_SnapObjects.size());
    for (ObjectIt o_it=m_SnapObjects.begin(); o_it!=m_SnapObjects.end(); o_it++)
    	F.w_stringZ	((*o_it)->GetName());
    F.close_chunk	();  
}
//----------------------------------------------------

void ESceneAIMapTool::SaveSelection(IWriter& F)
{
	SaveStream(F);
}

bool ESceneAIMapTool::Valid()
{
	return !m_Nodes.empty();
}

bool ESceneAIMapTool::IsNeedSave()
{
	return (!m_Nodes.empty()||!m_SnapObjects.empty());
}

void ESceneAIMapTool::OnObjectRemove(CCustomObject* O, bool bDeleting)
{
	if (OBJCLASS_SCENEOBJECT==O->FClassID){
    	if (find(m_SnapObjects.begin(),m_SnapObjects.end(),O)!=m_SnapObjects.end()){
			m_SnapObjects.remove(O);
	    	RealUpdateSnapList();
        }
    }
}

int ESceneAIMapTool::AddNode(const Fvector& pos, bool bIgnoreConstraints, bool bAutoLink, int sz)
{
   	Fvector Pos				= pos;
    if (1==sz)
    {
        SAINode* N 			= BuildNode(Pos,Pos,bIgnoreConstraints,true);
        if (N)
        {
            N->flags.set	(SAINode::flSelected,TRUE);
            if (bAutoLink) 	UpdateLinks(N,bIgnoreConstraints);
            return			1;
        }
        else
        {
            //ELog.Msg		(mtError,"Can't create node.");
            return 			0;
        }
    }
    else
    {
		return BuildNodes	(Pos,sz,bIgnoreConstraints);
    }
}

struct invalid_node_pred 
{
	int link;
	invalid_node_pred(int _link):link(_link){;}
	bool operator()(const SAINode*& x){ return x->Links()==link; }
};

void ESceneAIMapTool::SelectNodesByLink(int link)
{
    SelectObjects		(false);
    // remove link to sel nodes
    for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
        if ((*it)->Links()==link)
//			if (!(*it)->flags.is(SAINode::flHide))
	            (*it)->flags.set(SAINode::flSelected,TRUE);
    UI->RedrawScene		();
}

void ESceneAIMapTool::SelectObjects(bool flag)
{
    switch (LTools->GetSubTarget()){
    case estAIMapNode:{
        for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
//			if (!(*it)->flags.is(SAINode::flHide))
	            (*it)->flags.set(SAINode::flSelected,flag);
    }break;
    }
    UpdateHLSelected	();
    UI->RedrawScene		();
}


struct delete_sel_node_pred 
{
	bool operator()(SAINode*& x)
    {
    	// breaking links
        for (int k=0; k<4; k++)
            if (x->n[k]&&x->n[k]->flags.is(SAINode::flSelected))
                x->n[k]=0;
		// free memory                
    	bool res	= x->flags.is(SAINode::flSelected); 
        if (res) 	xr_delete(x); 
        return 		res; 
    }
};

void ESceneAIMapTool::RemoveSelection()
{
    switch (LTools->GetSubTarget()){
    case estAIMapNode:{
    	if (m_Nodes.size()==(u32)SelectionCount(true)){
        	Clear	(true);
        }else{
        	SPBItem* pb = UI->ProgressStart(3,"Removing nodes...");
        	// remove link to sel nodes
	        pb->Inc("erasing nodes");
            // remove sel nodes
           	AINodeIt result		= std::remove_if(m_Nodes.begin(), m_Nodes.end(), delete_sel_node_pred());
            m_Nodes.erase		(result,m_Nodes.end());
	        pb->Inc("updating hash");
            hash_Clear		   	();
		    hash_FillFromNodes 	();
	        pb->Inc("end");
            UI->ProgressEnd(pb);
        }
    }break;
    }
    UpdateHLSelected	();
    UI->RedrawScene		();
}

void ESceneAIMapTool::InvertSelection()
{
    switch (LTools->GetSubTarget()){
    case estAIMapNode:{
        for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
//			if (!(*it)->flags.is(SAINode::flHide))
	            (*it)->flags.invert(SAINode::flSelected);
    }break;
    }
    UpdateHLSelected	();
    UI->RedrawScene		();
}

int ESceneAIMapTool::SelectionCount(bool testflag)
{
	int count = 0;
    switch (LTools->GetSubTarget()){
    case estAIMapNode:{
        for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
            if ((*it)->flags.is(SAINode::flSelected)==testflag)
                count++;
    }break;
    }
    return count;
}

void ESceneAIMapTool::FillProp(LPCSTR pref, PropItemVec& items)
{
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Draw Nodes"),			&m_Flags, 		flHideNodes, 0,0, FlagValueCustom::flInvertedDraw);
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Slow Calculate Mode"),	&m_Flags, 		flSlowCalculate);
    PHelper().CreateFloat 	(items, PrepareKey(pref,"Common\\Visible Radius"),		&m_VisRadius, 	10.f, 	2500.f);
    PHelper().CreateFloat 	(items, PrepareKey(pref,"Common\\Smooth Height"),		&m_SmoothHeight,0.1f,	100.f);

    PHelper().CreateU32	 	(items, PrepareKey(pref,"Params\\Brush Size"),			&m_BrushSize, 	1, 100);
    PHelper().CreateFloat 	(items, PrepareKey(pref,"Params\\Can Up"),				&m_Params.fCanUP, 	0.f, 10.f);
    PHelper().CreateFloat 	(items, PrepareKey(pref,"Params\\Can Down"),			&m_Params.fCanDOWN, 0.f, 10.f);
}

void ESceneAIMapTool::GetBBox(Fbox& bb, bool bSelOnly)
{
    switch (LTools->GetSubTarget()){
    case estAIMapNode:{
    	if (bSelOnly){
            for (AINodeIt it=m_Nodes.begin(); it!=m_Nodes.end(); it++)
                if ((*it)->flags.is(SAINode::flSelected)){
                	bb.modify(Fvector().add((*it)->Pos,-m_Params.fPatchSize*0.5f));
                	bb.modify(Fvector().add((*it)->Pos,m_Params.fPatchSize*0.5f));
                }
        }else{
        	bb.merge		(m_AIBBox);
        }
    }break;
    }
}

