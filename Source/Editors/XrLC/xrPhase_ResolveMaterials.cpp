#include "stdafx.h"
#include "build.h"
#include "../xrLCLight/xrLC_GlobalData.h"
#include "../xrLCLight/xrface.h"

extern void		Detach		(xr_vector<Face*>* S);

struct _counter
{
	u16	dwMaterial;
	u32	dwCount;
};

// #include <ppl.h>
// #include <concurrent_vector.h>


void	CBuild::xrPhase_ResolveMaterials()
{
	/* CTimer  tProcecss; tProcecss.Start();

	// Count number of materials
	// Calculating materials
	auto& faces = lc_global_data()->g_faces();
	std::unordered_map<u16, size_t> matToIndex;

	// Локальные хранилища для потоков -> потом сведём в общий map
	concurrency::combinable<std::unordered_map<u16, u32>> localCounts;

	Concurrency::parallel_for_each(faces.begin(), faces.end(), [&localCounts](Face* F)
	{
		localCounts.local()[F->dwMaterial] += 1;
	});

	// Слияние локальных карт в глобальную
	std::unordered_map<u16, u32> globalCounts;
	localCounts.combine_each(
		[&globalCounts](const std::unordered_map<u16, u32>& lm)
		{
			for (const auto& kv : lm)
				globalCounts[kv.first] += kv.second;
		});


	// ======================================================
	// 2) Вектор счётчиков + карта material -> index (SC)
	// ======================================================
	xr_vector<_counter> count;
	count.reserve(globalCounts.size());
	matToIndex.reserve(globalCounts.size());

	size_t idx = 0;
	for (const auto& kv : globalCounts)
	{
		const u16 mat = kv.first;
		const u32 cnt = kv.second;
		count.push_back(_counter{ mat, cnt });
		matToIndex[mat] = idx++;
	}

	// Performing Subdivs
	concurrency::concurrent_vector<concurrency::concurrent_vector<Face*>> bins;
	bins.reserve(count.size());
	bins.resize(count.size());
	concurrency::parallel_for_each(faces.begin(), faces.end(), [&matToIndex, &bins](Face* F)
	{
		if (!F->Shader().flags.bRendering) return;

		auto it = matToIndex.find(F->dwMaterial);
		if (it != matToIndex.end())
		{
			bins[it->second].push_back(F);
		}
	});

	// Переносим в итоговый g_XSplit
	g_XSplit.reserve(count.size());
	g_XSplit.resize(count.size());

	for (size_t i = 0; i < g_XSplit.size(); ++i)
	{
		// vecFace имеет конструктор от итераторов
		g_XSplit[i] = new xr_vector<Face*>(bins[i].begin(), bins[i].end());
	}

	// Старый код
	{
		for (int SP = 0; SP<int(g_XSplit.size()); SP++)
		{
			if (g_XSplit[SP]->empty())
				xr_delete(g_XSplit[SP]);
		}

		g_XSplit.erase(std::remove_if(g_XSplit.begin(), g_XSplit.end(), [](xr_vector<Face*>* a) { return a == nullptr; }), g_XSplit.end());
	} 

	for (auto F : g_XSplit)
		Detach(F);

	clMsg("Material %u subdivisions. %u ms", g_XSplit.size(), tProcecss.GetElapsed_ms());
	*/

	// Count number of materials
	Status("Calculating materials/subdivs...");
	xr_vector<_counter>	counts;
	{
		counts.reserve(256);
		for (auto F_it = lc_global_data()->g_faces().begin(); F_it != lc_global_data()->g_faces().end(); F_it++)
		{
			Face* F = *F_it;
			BOOL	bCreate = TRUE;
			for (u32 I = 0; I < counts.size(); I++)
			{
				if (F->dwMaterial == counts[I].dwMaterial)
				{
					counts[I].dwCount += 1;
					bCreate = FALSE;
					break;
				}
			}
			if (bCreate) {
				_counter	C;
				C.dwMaterial = F->dwMaterial;
				C.dwCount = 1;
				counts.push_back(C);
			}
			Progress(float(F_it - lc_global_data()->g_faces().begin()) / float(lc_global_data()->g_faces().size()));
		}
	}

	Status("Perfroming subdivisions...");
	{
		g_XSplit.reserve(64 * 1024);
		g_XSplit.resize(counts.size());
		for (u32 I = 0; I < counts.size(); I++)
		{
			g_XSplit[I] = xr_new<xr_vector<Face*>>();
			g_XSplit[I]->reserve(counts[I].dwCount);
		}

		for (auto F_it = lc_global_data()->g_faces().begin(); F_it != lc_global_data()->g_faces().end(); F_it++)
		{
			Face* F = *F_it;
			if (!F->Shader().flags.bRendering)	continue;

			for (u32 I = 0; I < counts.size(); I++)
			{
				if (F->dwMaterial == counts[I].dwMaterial)
				{
					g_XSplit[I]->push_back(F);
				}
			}
			Progress(float(F_it - lc_global_data()->g_faces().begin()) / float(lc_global_data()->g_faces().size()));
		}
	}

	Status("Removing empty subdivs...");
	{
		for (int SP = 0; SP<int(g_XSplit.size()); SP++)
			if (g_XSplit[SP]->empty())	xr_delete(g_XSplit[SP]);
		g_XSplit.erase(std::remove(g_XSplit.begin(), g_XSplit.end(), (xr_vector<Face*>*)NULL), g_XSplit.end());
	}

	Status("Detaching subdivs...");
	{
		for (u32 it = 0; it < g_XSplit.size(); it++)
		{
			Detach(g_XSplit[it]);
		}
	}
	clMsg("%d subdivisions.", g_XSplit.size());
}
