#include "stdafx.h"
#include "build.h"
#include "../xrLCLight/xrdeflector.h"
#include "../xrLCLight/xrface.h"
#include "../xrLCLight/xrlc_globaldata.h"

extern void		Detach		(xr_vector<Face*>* S);
 
namespace
{
    void setup_bbs(Fbox& b1, Fbox& b2, const Fbox& bb, int edge)
    {
        Fvector size;
        size.sub(bb.max, bb.min);

        b1 = bb;
        b2 = bb;

        switch (edge)
        {
        case 0: b1.max.x -= size.x / 2; b2.min.x += size.x / 2; break;
        case 1: b1.max.y -= size.y / 2; b2.min.y += size.y / 2; break;
        case 2: b1.max.z -= size.z / 2; b2.min.z += size.z / 2; break;
        }
    }

    bool should_split(const xr_vector<Face*>& faces, const Fbox& bb)
    {
        if ((int)faces.size() < c_SS_LowVertLimit * 2)
            return false;

        Fvector size;
        size.sub(bb.max, bb.min);

        if (size.x > c_SS_maxsize || size.y > c_SS_maxsize || size.z > c_SS_maxsize)
            return true;

        if ((int)faces.size() > c_SS_HighVertLimit)
            return true;

        auto* defl = (CDeflector*)faces.front()->pDeflector;
        if (defl)
        {
            u8 BORDER = gCompilerMode.LC_lmap_BORDER;

            if (defl->layer.width >= (gCompilerMode.LC_lmap_size - 2 * BORDER) ||
                defl->layer.height >= (gCompilerMode.LC_lmap_size - 2 * BORDER))
            {
                clMsg("Split: Deflector size %u x %u exceeds limits",
                    defl->layer.width, defl->layer.height);
                return true;
            }
        }
        return false;
    }

    int select_longest_axis(const Fvector& size)
    {
        if (size.x >= size.y && size.x >= size.z) return 0;
        if (size.y >= size.x && size.y >= size.z) return 1;
        return 2;
    }

    void split_faces(const xr_vector<Face*>& source, const Fbox& b1, xr_vector<Face*>& out1, xr_vector<Face*>& out2)
    {
        out1.clear();
        out2.clear();
        out1.reserve(source.size());
        out2.reserve(source.size());

        for (auto* face : source)
        {
            Fvector center;
            face->CalcCenter(center);
            (b1.contains(center) ? out1 : out2).push_back(face);
        }
    }

    void create_deflectors(xr_vector<Face*>& s1, xr_vector<Face*>& s2)
    {
        auto& deflectors = lc_global_data()->g_deflectors();

        auto make_deflector = [&](xr_vector<Face*>& faces)
            {
                CDeflector* D = new CDeflector();
                D->OA_Place(faces);
                D->OA_Export();
                deflectors.push_back(D);
            };

        make_deflector(s1);
        make_deflector(s2);
    }

    void remove_deflector(CDeflector* defl)
    {
        auto& deflectors = lc_global_data()->g_deflectors();
        auto it = std::find(deflectors.begin(), deflectors.end(), defl);
        if (it != deflectors.end())
        {
            xr_delete(*it);
            deflectors.erase(it);
        }
    }
}

void CBuild::xrPhase_Subdivide()
{
    Status("Subdividing in space...");

    for (size_t X = 0; X < g_XSplit.size();)
    {
        auto& faces = *g_XSplit[X];

        if (faces.empty())
        {
            xr_delete(g_XSplit[X]);
            g_XSplit.erase(g_XSplit.begin() + X);
            continue;
        }

        Progress(float(X) / g_XSplit.size());
        AditionalData("Splitting: %u | %u", X, g_XSplit.size());

        // Bounding box
        Fbox bb;
        bb.invalidate();
        for (auto* f : faces)
        {
            bb.modify(f->v[0]->P);
            bb.modify(f->v[1]->P);
            bb.modify(f->v[2]->P);
        }

        if (!should_split(faces, bb))
        {
            ++X;
            continue;
        }

        // Longest axis split
        Fvector size;
        size.sub(bb.max, bb.min);
        int axis = select_longest_axis(size);

        Fbox b1, b2;
        setup_bbs(b1, b2, bb, axis);

        xr_vector<Face*> s1, s2;
        split_faces(faces, b1, s1, s2);

        // If split failed, skip
        if (s1.size() < c_SS_LowVertLimit || s2.size() < c_SS_LowVertLimit)
        {
            ++X;
            continue;
        }

        // Split deflector
        if (auto* defl_base = (CDeflector*)faces.front()->pDeflector)
        {
            remove_deflector(defl_base);
            create_deflectors(s1, s2);
        }

        // Replace with two splits
        xr_delete(g_XSplit[X]);
        g_XSplit.erase(g_XSplit.begin() + X);
        g_XSplit.push_back(new xr_vector<Face*>(std::move(s1)));
        g_XSplit.push_back(new xr_vector<Face*>(std::move(s2)));
        Detach(&s1);
        Detach(&s2);
    }

    clMsg("%d subdivisions.", g_XSplit.size());
    validate_splits();

    size_t allocated = 0;
    for (auto* D : lc_global_data()->g_deflectors())
        allocated += D->size_deflector();

    allocated /= (1024 * 1024);
    AditionalData("Splits: %u | DeflectorsAlloc: %u mb", g_XSplit.size(), allocated);
}