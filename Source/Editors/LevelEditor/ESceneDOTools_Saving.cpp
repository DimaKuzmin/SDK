#include "stdafx.h"
#include "ESceneDOTools.h"
  
static const u32 DETMGR_VERSION = 0x0003ul;

// Color Indexes
void EDetailManager::SaveColorIndices(IWriter& F)
{
    // objects
    F.open_chunk(DETMGR_CHUNK_OBJECTS);
    for (DetailIt it = objects.begin(); it != objects.end(); it++)
    {
        F.open_chunk(it - objects.begin());
        ((EDetail*)(*it))->Save(F);
        F.close_chunk();
    }
    F.close_chunk();
    // color index map
    F.open_chunk(DETMGR_CHUNK_COLOR_INDEX);
    F.w_u8((u8)m_ColorIndices.size());
    ColorIndexPairIt S = m_ColorIndices.begin();
    ColorIndexPairIt E = m_ColorIndices.end();
    ColorIndexPairIt i_it = S;
    for (; i_it != E; i_it++)
    {
        F.w_u32(i_it->first);
        F.w_u8((u8)i_it->second.size());
        for (DOIt d_it = i_it->second.begin(); d_it != i_it->second.end(); d_it++)
            F.w_stringZ((*d_it)->GetName());
    }
    F.close_chunk();
}

bool EDetailManager::LoadColorIndices(IReader& F)
{
    bool bRes = true;
    // objects
    IReader* OBJ = F.open_chunk(DETMGR_CHUNK_OBJECTS);
    if (OBJ)
    {
        IReader* O = OBJ->open_chunk(0);
        for (int count = 1; O; count++) {
            EDetail* DO = xr_new<EDetail>();
            if (DO->Load(*O)) 	objects.push_back(DO);
            else				bRes = false;
            O->close();
            O = OBJ->open_chunk(count);
        }
        OBJ->close();
    }
    // color index map
    R_ASSERT(F.find_chunk(DETMGR_CHUNK_COLOR_INDEX));
    int cnt = F.r_u8();
    string256			buf;
    u32 index;
    int ref_cnt;
    for (int k = 0; k < cnt; k++)
    {
        index = F.r_u32();
        ref_cnt = F.r_u8();
        for (int j = 0; j < ref_cnt; j++)
        {
            F.r_stringZ(buf, sizeof(buf));
            EDetail* DO = FindDOByName(buf);
            if (DO) 	m_ColorIndices[index].push_back(DO);
            else		bRes = false;
        }
    }
    InvalidateCache();

    return bRes;
}

// Save Selection, Stream

bool EDetailManager::LoadStream(IReader& F)
{
    inherited::LoadStream(F);

    string256 buf;
    R_ASSERT(F.find_chunk(DETMGR_CHUNK_VERSION));
    u32 version = F.r_u32();

    if (version != DETMGR_VERSION) {
        ELog.Msg(mtError, "EDetailManager: unsupported version.");
        return false;
    }

    if (F.find_chunk(DETMGR_CHUNK_FLAGS)) m_Flags.assign(F.r_u32());

    // header
    R_ASSERT(F.r_chunk(DETMGR_CHUNK_HEADER, &dtH));

    // slots
    R_ASSERT(F.find_chunk(DETMGR_CHUNK_SLOTS));
    int slot_cnt = F.r_u32();
    if (slot_cnt)
        dtSlots = xr_alloc<DetailSlot>(slot_cnt);

    m_Selected.resize(slot_cnt);
    F.r(dtSlots, slot_cnt * sizeof(DetailSlot));

    // objects
    if (!LoadColorIndices(F))
    {
        ELog.DlgMsg(mtError, "EDetailManager: Some objects removed. Reinitialize objects.", buf);
        InvalidateSlots();
    }

    // internal bbox
    R_ASSERT(F.r_chunk(DETMGR_CHUNK_BBOX, &m_BBox));

    // snap objects
    if (F.find_chunk(DETMGR_CHUNK_SNAP_OBJECTS)) {
        int snap_cnt = F.r_u32();
        if (snap_cnt) {
            for (int i = 0; i < snap_cnt; i++) {
                F.r_stringZ(buf, sizeof(buf));
                CCustomObject* O = Scene->FindObjectByName(buf, OBJCLASS_SCENEOBJECT);
                if (!O)		ELog.Msg(mtError, "EDetailManager: Can't find snap object '%s'.", buf);
                else		m_SnapObjects.push_back(O);
            }
        }
    }

    if (F.find_chunk(DETMGR_CHUNK_DENSITY))
        ps_r__Detail_density = F.r_float();

    // base texture
    if (F.find_chunk(DETMGR_CHUNK_BASE_TEXTURE))
    {
        F.r_stringZ(buf, sizeof(buf));
        if (m_Base.LoadImage(buf))
        {
            m_Base.CreateShader();
            m_RTFlags.set(flRTGenerateBaseMesh, TRUE);
        }
        else {
            ELog.Msg(mtError, "EDetailManager: Can't find base texture '%s'.", buf);
            ClearSlots();
            ClearBase();
        }
    }

    InvalidateCache();

    return true;
}

bool EDetailManager::LoadSelection(IReader& F)
{
    Clear();
    return LoadStream(F);
}

void EDetailManager::SaveStream(IWriter& F)
{
    inherited::SaveStream(F);

    // version
    F.open_chunk(DETMGR_CHUNK_VERSION);
    F.w_u32(DETMGR_VERSION);
    F.close_chunk();

    F.open_chunk(DETMGR_CHUNK_FLAGS);
    F.w_u32(m_Flags.get());
    F.close_chunk();

    // header
    F.w_chunk(DETMGR_CHUNK_HEADER, &dtH, sizeof(DetailHeader));

    // objects
    SaveColorIndices(F);

    // slots
    F.open_chunk(DETMGR_CHUNK_SLOTS);
    F.w_u32(dtH.size_x * dtH.size_z);
    F.w(dtSlots, dtH.size_x * dtH.size_z * sizeof(DetailSlot));
    F.close_chunk();

    // internal bbox
    F.w_chunk(DETMGR_CHUNK_BBOX, &m_BBox, sizeof(Fbox));
    // base texture
    if (m_Base.Valid())
    {
        F.open_chunk(DETMGR_CHUNK_BASE_TEXTURE);
        F.w_stringZ(m_Base.GetName());
        F.close_chunk();
    }
    F.open_chunk(DETMGR_CHUNK_DENSITY);
    F.w_float(ps_r__Detail_density);
    F.close_chunk();
    // snap objects
    F.open_chunk(DETMGR_CHUNK_SNAP_OBJECTS);
    F.w_u32(m_SnapObjects.size());
    for (ObjectIt o_it = m_SnapObjects.begin(); o_it != m_SnapObjects.end(); o_it++)
        F.w_stringZ((*o_it)->GetName());
    F.close_chunk();
}

void EDetailManager::SaveSelection(IWriter& F)
{
    SaveStream(F);
}


// LTX Unsopported

bool EDetailManager::LoadColorIndicesLTX(CInifile& file)
{
    return false;
}
bool EDetailManager::LoadLTX(CInifile& ini)
{
    R_ASSERT2(0, "not_implemented");
    return true;
}


void EDetailManager::SaveColorIndicesLTX(CInifile& file)
{
    string_path path;
    xr_strcat(path, file.fname());
    xr_strcat(path, "_colors");

    CInifile* ini = xr_new<CInifile>(path);
    if (ini)
    {
        int i = 0;
        for (DetailIt it = objects.begin(); it != objects.end(); it++)
        {
            string32 name = { 0 };
            string32 tmp;

            xr_strcat(name, "detail_object_");
            xr_strcat(name, itoa(i, tmp, 10));

            ((EDetail*)(*it))->SaveLTX(*ini, name);;
        }

        i = 0;
        for (auto color_i : m_ColorIndices)
        {
            string32 name = { 0 };
            string32 tmp;

            xr_strcat(name, "detail_color_");
            xr_strcat(name, itoa(i, tmp, 10));

            ini->w_u32("", "", color_i.first);
            ini->w_u8("", "", (u8)color_i.second.size());

            int i_sec = 0;
            for (auto sec_i : color_i.second)
            {
                string32 name_second = { 0 };
                xr_strcat(name_second, "sec_");
                xr_strcat(name_second, itoa(i_sec, name, 10));

                ini->w_string(name, name_second, sec_i->GetName());
            }
        }
    }
    ini->save_as(path);
}

void EDetailManager::SaveLTX(CInifile& ini, int id)
{
    int slot_cnt = dtH.size_x * dtH.size_z;
    LPCSTR name = ini.fname();
    string128 path;
    xr_strcat(path, name);
    xr_strcat(path, "_src");

    CInifile* file_ = xr_new<CInifile>(path, false, false, false);
    CInifile file = *file_;

    for (int slot_idx = 0; slot_idx < slot_cnt; slot_idx++)
    {
        DetailSlot* it = &dtSlots[slot_idx];

        string128 name = { 0 };
        string32 tmp = { 0 };
        xr_strcat(name, "dt_slot_");
        xr_strcat(name, itoa(slot_idx, tmp, 10));

        file.w_u32(name, "id0", it->id0);
        file.w_u32(name, "id1", it->id1);
        file.w_u32(name, "id2", it->id2);
        file.w_u32(name, "id3", it->id3);

        file.w_u32(name, "blue", it->c_b);
        file.w_u32(name, "red", it->c_r);
        file.w_u32(name, "green", it->c_g);

        file.w_u32(name, "dir", it->c_dir);
        file.w_u32(name, "hemi", it->c_hemi);

        file.w_u32(name, "y_base", it->y_base);
        file.w_u32(name, "y_height", it->y_height);
    }

    file.save_as(path);
}