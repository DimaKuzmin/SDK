#include "stdafx.h"
#include "ESceneDOTools.h"
extern void bwdithermap(int levels, int magic[16][16]);

//------------------------------------------------------------------------------
EDetailManager::EDetailManager():ESceneToolBase(OBJCLASS_DO)
{
	dtSlots				= 0;
    ZeroMemory			(&dtH,sizeof(dtH));
    m_Selected.clear	();
    InitRender			();
    m_Flags.assign		(flObjectsDraw);
}

EDetailManager::~EDetailManager(){
	Clear	();
    Unload	();
}
//------------------------------------------------------------------------------

void EDetailManager::ClearColorIndices()
{
	inherited::Clear	();
    RemoveDOs			();
    m_ColorIndices.clear();
}

void EDetailManager::ClearSlots()
{
    ZeroMemory			(&dtH,sizeof(DetailHeader));
    xr_free				(dtSlots);
	m_Selected.clear	();
    InvalidateCache		();
}

void EDetailManager::ClearBase()
{
    m_Base.Clear		();
    m_SnapObjects.clear	();
    ExecCommand			(COMMAND_REFRESH_SNAP_OBJECTS);
}

void EDetailManager::Clear(bool bSpecific)
{
	ClearBase			();
	ClearColorIndices	();
    ClearSlots			();
    m_Flags.zero		();
    m_RTFlags.zero		();
}
//------------------------------------------------------------------------------

void EDetailManager::InvalidateCache()
{
	// resize visible
	m_visibles[0].resize	(objects.size());	// dump(visible[0]);
	m_visibles[1].resize	(objects.size());	// dump(visible[1]);
	m_visibles[2].resize	(objects.size());	// dump(visible[2]);
	// Initialize 'vis' and 'cache'
	cache_Initialize	();
}


void EDetailManager::InitRender()
{
	// inavlidate cache
	InvalidateCache		();
	// Make dither matrix
	bwdithermap		(2,dither);

	soft_Load	();
}
//------------------------------------------------------------------------------

void EDetailManager::OnRender(int priority, bool strictB2F)
{
	if (dtSlots)
    {
    	if (1==priority){
        	if (false==strictB2F){
            	if (m_Flags.is(flSlotBoxesDraw)){
                    RCache.set_xform_world(Fidentity);
                    EDevice.SetShader	(EDevice.m_WireShader);

                    Fvector			c;
                    Fbox			bbox;
                    u32			inactive = 0xff808080;
                    u32			selected = 0xffffffff;
                    float dist_lim	= 75.f*75.f;
                    for (u32 z=0; z<dtH.size_z; z++){
                        c.z			= fromSlotZ(z);
                        for (u32 x=0; x<dtH.size_x; x++){
                            bool bSel 	= m_Selected[z*dtH.size_x+x];
                            DetailSlot* slot = dtSlots+z*dtH.size_x+x;
                            c.x			= fromSlotX(x);
                            c.y			= slot->r_ybase()+slot->r_yheight()*0.5f; //(slot->y_max+slot->y_min)*0.5f;
                            float dist = EDevice.m_Camera.GetPosition().distance_to_sqr(c);
                         	if ((dist<dist_lim)&&::Render->ViewBase.testSphere_dirty(c,DETAIL_SLOT_SIZE_2)){
								bbox.min.set(c.x-DETAIL_SLOT_SIZE_2, slot->r_ybase(), 					c.z-DETAIL_SLOT_SIZE_2);
                            	bbox.max.set(c.x+DETAIL_SLOT_SIZE_2, slot->r_ybase()+slot->r_yheight(),	c.z+DETAIL_SLOT_SIZE_2);
                            	bbox.shrink	(0.05f);
								DU_impl.DrawSelectionBoxB(bbox,bSel?&selected:&inactive);
							}
                        }
                    }
                }
            }else{
				RCache.set_xform_world				(Fidentity);
                if (m_Flags.is(flBaseTextureDraw))	m_Base.Render			(m_Flags.is(flBaseTextureBlended));
				if (m_Flags.is(flObjectsDraw))		CDetailManager::Render	();
            }
        }
    }
}
//------------------------------------------------------------------------------

void EDetailManager::OnDeviceCreate()
{
	// base texture
    m_Base.CreateShader();
	// detail objects
	for (DetailIt it=objects.begin(); it!=objects.end(); it++)
    	((EDetail*)(*it))->OnDeviceCreate();
	soft_Load	();
}

void EDetailManager::OnDeviceDestroy()
{
	// base texture
    m_Base.DestroyShader();
	// detail objects
	for (DetailIt it=objects.begin(); it!=objects.end(); it++)
    	((EDetail*)(*it))->OnDeviceDestroy();
	soft_Unload	();
}


void EDetailManager::OnObjectRemove(CCustomObject* O, bool bDeleting)
{
	ObjectIt it=std::find(m_SnapObjects.begin(),m_SnapObjects.end(),O);
	if (it!=m_SnapObjects.end()){
    	m_RTFlags.set		(flRTGenerateBaseMesh,TRUE);
		m_SnapObjects.remove(O);
    }
}

void EDetailManager::OnSynchronize()
{
}
void EDetailManager::OnSceneUpdate()       
{
}

void EDetailManager::OnFrame()
{
    if (m_RTFlags.is(flRTGenerateBaseMesh)&&m_Base.Valid())
    {
    	m_RTFlags.set		(flRTGenerateBaseMesh,FALSE);
	    m_Base.CreateRMFromObjects(m_BBox,m_SnapObjects);
    }
}

void EDetailManager::ExportColorIndices(LPCSTR fname)
{
	IWriter* F 	= FS.w_open(fname);
    if (F)
    {
	    SaveColorIndices(*F);
    	FS.w_close	(F);
    }
}

bool EDetailManager::ImportColorIndices(LPCSTR fname)
{
	IReader* F=FS.r_open(fname);
    if (F)
    {
        ClearColorIndices	();
        LoadColorIndices	(*F);
        FS.r_close			(F);
        return true;
    }else{
    	ELog.DlgMsg			(mtError,"Can't open file '%s'.",fname);
        return false;
    }
}

static const u32 DETMGR_VERSION = 0x0003ul;
bool EDetailManager::Export(LPCSTR path) 
{
    xr_string fn		= xr_string(path)+"build.details";
    bool bRes=true;

    R_ASSERT("DETAIL NOT SELECTED LIST", objects.size());

    if (!objects.size())
    {
        ELog.DlgMsg(mtError, "EDetailManager: No Selected Objects for Details...");
        return false;
    }


    SPBItem* pb = UI->ProgressStart(5,"Making details...");
	CMemoryWriter F;

    pb->Inc				("merge textures");
    Fvector2Vec			offsets;
    Fvector2Vec			scales;
    boolVec				rotated;
    RStringSet 			textures_set;
    RStringVec 			textures;
    U32Vec				remap;
    U8Vec remap_object	(objects.size(),u8(-1));

    int slot_cnt		= dtH.size_x*dtH.size_z;
	for (int slot_idx=0; slot_idx<slot_cnt; slot_idx++)
    {
    	DetailSlot* it 	= &dtSlots[slot_idx];
        for (int part=0; part<4; part++)
        {
        	u8 id		= it->r_id(part);
        	if (id!=DetailSlot::ID_Empty)
            {
            	textures_set.insert(((EDetail*)(objects[id]))->GetTextureName());
                remap_object[id] = 1;
            }
        }
    }
    textures.assign		(textures_set.begin(),textures_set.end());

    U8It remap_object_it= remap_object.begin();

    u32 new_idx			= 0;
    for (DetailIt d_it=objects.begin(); d_it!=objects.end(); d_it++,remap_object_it++)
    	if ((*remap_object_it==1)&&(textures_set.find(((EDetail*)(*d_it))->GetTextureName())!=textures_set.end()))
	    	*remap_object_it	= (u8)new_idx++;

    xr_string 			do_tex_name = ChangeFileExt(fn,"_details");
    int res				= ImageLib.CreateMergedTexture(
        textures,do_tex_name.c_str(),
        STextureParams::tfDXT5,  // se7kills ONLY DXT 1
        256, 4096, // X max
        256, 4096, // Y max
        offsets, scales,rotated,remap);
   
    if (1!=res)		
        bRes=FALSE;

    if (!bRes)
    {
        Msg("Can't Create Merged Texture!!!");
        UI->ProgressEnd(pb);
        ELog.DlgMsg(mtError, "EDetailManager Cant Create Merged Texture Size: %u | %u (engine hardkoded)", 4096, 4096);
        return false;
    }



    // objects
    int object_idx		= 0;
    if (bRes)
    {
        pb->Inc("export geometry");

	    do_tex_name 	= EFS.ExtractFileName(do_tex_name.c_str());
        F.open_chunk	(DETMGR_CHUNK_OBJECTS);
        for (DetailIt it=objects.begin(); it!=objects.end(); it++){
        	if (remap_object[it-objects.begin()]!=u8(-1)){
                F.open_chunk	(object_idx++);
                if (!((EDetail*)(*it))->m_pRefs){
                    ELog.DlgMsg(mtError, "Bad object or object not found '%s'.", ((EDetail*)(*it))->m_sRefs.c_str());
                    bRes=false;
                }else{
                    LPCSTR tex_name = ((EDetail*)(*it))->GetTextureName();
                    u32 t_idx = 0;
                    for (; t_idx<textures.size(); t_idx++) 
                        if (textures[t_idx]==tex_name) break;
                    VERIFY(t_idx<textures.size());
                    t_idx = remap[t_idx];
                    ((EDetail*)(*it))->Export	(F,do_tex_name.c_str(),offsets[t_idx],scales[t_idx],rotated[t_idx]);
                }
                F.close_chunk	();
                if (!bRes) break;
            }
        }
        F.close_chunk		();
    }
   
    if (!bRes)
        Msg("Can't export Geometry !!!");
  
   
    // slots
    if (bRes)
    {
        pb->Inc("export slots");

    	xr_vector<DetailSlot> dt_slots(slot_cnt);
        dt_slots.assign(dtSlots,dtSlots+slot_cnt);
        for (int slot_idx=0; slot_idx<slot_cnt; slot_idx++)
        {
            DetailSlot& it 	= dt_slots[slot_idx];
            // zero colors need lighting
	        it.c_dir		= 0;
	        it.c_hemi		= 0;
	        it.c_r			= 0;
	        it.c_g			= 0;
	        it.c_b			= 0;
            for (int part=0; part<4; part++)
            {
                u8 id		= it.r_id(part);
                if (id!=DetailSlot::ID_Empty)
                    it.w_id(part,remap_object[id]);
            }
        }
		F.open_chunk	(DETMGR_CHUNK_SLOTS);
		F.w				(dt_slots.data(),dtH.size_x*dtH.size_z*sizeof(DetailSlot));
	    F.close_chunk	();
        pb->Inc();

        // write header
        dtH.version		= DETAIL_VERSION;
        dtH.object_count= object_idx;

        F.w_chunk		(DETMGR_CHUNK_HEADER,&dtH,sizeof(DetailHeader));

    	bRes 			= F.save_to(fn.c_str());
    }

    if (!bRes)
        Msg("Can't Export Slots !!!");

    pb->Inc();
    UI->ProgressEnd(pb);
    return bRes;
}

void EDetailManager::OnDensityChange(PropValue* prop)
{
	InvalidateCache		();
}	
 
void EDetailManager::OnBaseTextureChange(PropValue* prop)
{
	m_Base.OnImageChange	(prop);
    InvalidateSlots			();
    ELog.DlgMsg				(mtInformation,"Texture changed. Reinitialize objects.");
}

void EDetailManager::FillProp(LPCSTR pref, PropItemVec& items)
{
	PropValue* P;
    P=PHelper().CreateFloat	(items, PrepareKey(pref,"Objects per square"),				&ps_r__Detail_density);
    P->OnChangeEvent.bind	(this,&EDetailManager::OnDensityChange);
    P=PHelper().CreateChoose(items, PrepareKey(pref,"Base Texture"),					&m_Base.name, smTexture);
    P->OnChangeEvent.bind	(this,&EDetailManager::OnBaseTextureChange);
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Draw objects"),			&m_Flags,	flObjectsDraw);
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Draw base texture"),		&m_Flags,	flBaseTextureDraw);
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Base texture blended"),	&m_Flags,	flBaseTextureBlended);
    PHelper().CreateFlag32	(items, PrepareKey(pref,"Common\\Draw slot boxes"),			&m_Flags,	flSlotBoxesDraw);
}

bool EDetailManager::GetSummaryInfo(SSceneSummary* inf)
{
	for (DetailIt it=objects.begin(); it!=objects.end(); it++){
    	((EDetail*)(*it))->OnDeviceCreate();
        CEditableObject* E 	= ((EDetail*)(*it))->m_pRefs;
		if (!E)				continue;
	    CSurface* surf		= *E->FirstSurface(); VERIFY(surf);
		inf->AppendTexture	(surf->_Texture(),SSceneSummary::sttDO,0,0,"$DETAILS$");
    }
    return true;
}

