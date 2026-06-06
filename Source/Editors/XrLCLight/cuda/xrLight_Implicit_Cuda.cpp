#include "..\stdafx.h"
#include "..\xrLight_ImplicitDeflector.h"

#include "../xrDeflector.h"
#include "../light_point.h"

// cuda
#include "xrDeflectorLight_Packed.h"


class CImplicitDeflector
{
public:
	ImplicitDeflector* defl = nullptr;

	CImplicitDeflector()
	{
	}

	void ApplyColor(size_t IndexTask, base_color_c& Cnew)
	{
		u32 U = GPUTaskinSystem.GetU(IndexTask);
		u32 V = GPUTaskinSystem.GetV(IndexTask);

		base_color_c Cadd;
		defl->Lumel(U, V)._get(Cadd); Cadd.add(Cnew);
		defl->Lumel(U, V)._set(Cadd);
		defl->Samples(U, V) += 1;
	}

	void ApplyColors()
	{
		for (auto V = 0; V < defl->Height(); V++)
			for (auto U = 0; U < defl->Width(); U++)
			{
				u8 samples = defl->Samples(U, V);

				if (samples)
				{
					base_color_c cAdd;
					defl->Lumel(U, V)._get(cAdd);
					cAdd.scale(samples);
					cAdd.mul(0.5f);
					defl->Lumel(U, V)._set(cAdd);
					defl->Marker(U, V) = 255;
				}
				else
				{
					defl->Marker(U, V) = 0;
				}
			}
	}

	void RunTaskGPU()
	{
		clMsg("$ Run Tasks GPU");
		defl = &cl_globs.DATA();

		CTimer tStats;
		tStats.Start();

		GPUTaskinSystem.RestartALL();

		u32 flags = (gCompilerMode.LC_NoSun ? LP_dont_sun : 0);
		GPUTaskinSystem.current_flags = flags;
		GPUTaskinSystem.ColorsMapType = eImplicit;

		// Однопоточный режим пока что 
		static std::atomic<u32> task_height = 0;
		task_height = 0;

		auto ProcessLmap = []( )
		{
			auto defl = &cl_globs.DATA();

			// Setup variables
			Fvector2 dim, half;
			dim.set(float(defl->Width()), float(defl->Height()));
			half.set(.5f / dim.x, .5f / dim.y);

			// Jitter data
			Fvector2 JS;
			JS.set(.499f / dim.x, .499f / dim.y);
			u32 Jcount;
			Fvector2* Jitter;
			Jitter_Select(Jitter, Jcount);

			while (true)
			{
				auto V = task_height.fetch_add(1);
				if (V >= defl->Height()) break;

				for (u32 U = 0; U < defl->Width(); U++)
				{
					try
					{
						for (u32 SampleID = 0; SampleID < Jcount; SampleID++)
						{
							// LUMEL space
							Fvector2				P;
							P.x = float(U) / dim.x + half.x + Jitter[SampleID].x * JS.x;
							P.y = float(V) / dim.y + half.y + Jitter[SampleID].y * JS.y;

							// World space
							Fvector wP, wN, B;
							for (auto F : cl_globs.query(P.x, P.y))
							{
								_TCF& tc = F->tc[0];
								if (tc.isInside(P, B))
								{
									// We found triangle and have barycentric coords
									Vertex* V1 = F->v[0];
									Vertex* V2 = F->v[1];
									Vertex* V3 = F->v[2];
									wP.from_bary(V1->P, V2->P, V3->P, B);
									wN.from_bary(V1->N, V2->N, V3->N, B);
									wN.normalize();

									GPUTaskinSystem.LightPointPacked_add_task(GPUTaskinSystem.MakeKey(U, V), nullptr, wP, wN, F);
								}
							}
						}
					}
					catch (...)
					{
						clMsg("* THREAD #%d: Access violation. Possibly recovered.");//,thID
					}
				}

				AditionalData("Current: %u", V);
			};

			// Остаток доработать 
			GPUTaskinSystem.LightPointPacked_run_tasks();
		};
 
		runThreadsMax(ProcessLmap, gCompilerMode.ThreadsNum);

		ApplyColors();
 		GPUTaskinSystem.RestartALL();
	}
};

CImplicitDeflector GPU_DeflectorIMPL;
void ApplyColorGPU(size_t IndexTask, base_color_c& C)
{
	GPU_DeflectorIMPL.ApplyColor(IndexTask, C);
}

void RunImplicitCuda()
{
	// Start threads
	GPU_DeflectorIMPL.RunTaskGPU();
}