#pragma once

#include "../XrECore/Editor/face_smoth_flags.h"
// #include "FaceComponentSplitter.h"
#include "itterate_adjacents_static.h"



template	<typename typeVertex>
class calculate_normals
{

	typedef	typeVertex											type_vertex;
	typedef	calculate_normals<typeVertex>						type_self;
	typedef	typename typeVertex::type_face						type_face;
	//these typedefs to hide global typedefs!!!
	typedef xr_vector<type_vertex*>								vecVertex;
	typedef typename vecVertex::iterator						vecVertexIt;
	typedef xr_vector<type_face*>								vecFace;
	typedef typename vecFace::iterator							vecFaceIt;

	typedef vecFace												vecAdj;
	typedef typename vecAdj::iterator							vecAdjIt;

private:
    typedef  itterate_adjacents< itterate_adjacents_params_static<type_vertex> > itterate_adjacents_type;

public:
	static void	calc_normals(vecVertex& vertices, vecFace& faces)
	{
        u32 Vcount = vertices.size();
        float sm_cos = _cos(deg2rad(g_params().m_sm_angle));

        // ----------------------------------------------------
        // 1. Сбрасываем и считаем нормали у всех граней
        // ----------------------------------------------------
        for (auto F : faces)
        {
			F->flags.bSplitted = false;
			F->CalcNormal();
        }
 
  		for (u32 I = 0; I < Vcount; I++)
		{
 			type_vertex* pTestVertex = vertices[I];
			for (auto& F : pTestVertex->m_adjacents)
 				F->flags.bSplitted = false;
 
			std::sort(pTestVertex->m_adjacents.begin(), pTestVertex->m_adjacents.end());
			while (pTestVertex->m_adjacents.size())
			{
				vecFace new_adj;
				itterate_adjacents_type::recurse_tri_params p(pTestVertex, new_adj, sm_cos);
				itterate_adjacents_type::RecurseTri(0, p);

				type_vertex* pNewVertex = pTestVertex->CreateCopy_NOADJ(vertices);

				for (u32 a = 0; a < new_adj.size(); ++a)
				{
					type_face* test = new_adj[a];
					test->VReplace(pTestVertex, pNewVertex);
				}

				pNewVertex->normalFromAdj();
			}
		}

        // for (u32 I = 0; I < Vcount; I++)
        // {
        //     // Фигня нагружает 
        //     type_vertex* pTestVertex = vertices[I];
        // 
        //     FaceComponentSplitter<type_vertex> splitter(pTestVertex, sm_cos);
        //     xr_vector<xr_vector<type_face*>> components;
        //     splitter.BuildComponents(components);
        // 
        //     for (auto& comp : components)
        //     {
        //         type_vertex* newV = pTestVertex->CreateCopy_NOADJ(vertices);
        // 
        //         for (type_face* f : comp)
        //             f->VReplace(pTestVertex, newV);
        // 
        //         newV->normalFromAdj();
        //     }
        // }

		// Destroy unused vertices

		isolate_vertices<type_vertex>(FALSE, vertices);

		// Recalculate normals
		for (vecVertexIt it = vertices.begin(); it != vertices.end(); it++)
			(*it)->normalFromAdj();

		// clMsg("%d vertices was duplicated 'cause of SM groups", vertices.size() - Vcount);

		// Clear temporary flag
		for (vecFaceIt it = faces.begin(); it != faces.end(); it++)
			(*it)->flags.bSplitted = false;
	}
};
