#pragma once

template<typename TVertex>
class FaceComponentSplitter
{
public:
    using Vertex = TVertex;
    using Face = typename Vertex::type_face;
    using FaceVec = xr_vector<Face*>;

private:
    const Vertex* vertex = nullptr;
    float         smoothCos = 0.f;

    xr_vector<char> visited; // вместо bSplitted (локально, безопасно)

public:
    FaceComponentSplitter(const Vertex* v, float smCos) : vertex(v), smoothCos(smCos)
    {
    }

    // =========================================================
    // MAIN ENTRY
    // =========================================================
    void BuildComponents(xr_vector<FaceVec>& outComponents)
    {
        const auto& adj = vertex->m_adjacents;
        const size_t N = adj.size();

        visited.assign(N, 0);

        for (size_t i = 0; i < N; ++i)
        {
            if (visited[i])
                continue;

            FaceVec component;
            FloodFill(i, component);

            if (!component.empty())
                outComponents.push_back(std::move(component));
        }
    }

private:

    // =========================================================
    // ITERATIVE BFS/DFS (NO RECURSION)
    // =========================================================
    void FloodFill(size_t startIdx, FaceVec& out)
    {
        const auto& adj = vertex->m_adjacents;

        xr_vector<size_t> stack;
        stack.reserve(adj.size());

        stack.push_back(startIdx);
        visited[startIdx] = 1;

        while (!stack.empty())
        {
            size_t idx = stack.back();
            stack.pop_back();

            Face* f = adj[idx];
            out.push_back(f);

            // check all neighbors
            for (size_t j = 0; j < adj.size(); ++j)
            {
                if (visited[j])
                    continue;

                Face* g = adj[j];

                if (CanConnect(f, g))
                {
                    visited[j] = 1;
                    stack.push_back(j);
                }
            }
        }
    }

    // =========================================================
    // EDGE + SMOOTH CHECK (быстрая версия)
    // =========================================================
    bool CanConnect(const Face* a, const Face* b) const
    {
        // shared edge check (3x3, но без лишних копий)
        for (int i = 0; i < 3; ++i)
        {
            Vertex* a0, * a1;
            a->EdgeVerts(i, &a0, &a1);
            if (a0 > a1) std::swap(a0, a1);

            for (int j = 0; j < 3; ++j)
            {
                Vertex* b0, * b1;
                b->EdgeVerts(j, &b0, &b1);
                if (b0 > b1) std::swap(b0, b1);

                if (a0 == b0 && a1 == b1)
                {
                    // smoothing
                    if (!gCompilerMode.LC_NoSMG)
                    {
                        return do_connect_faces_by_faces_edge_flags(a->sm_group, b->sm_group, i, j);
                    }

                    return (a->N.dotproduct(b->N) > smoothCos);
                }
            }
        }

        return false;
    }
};
