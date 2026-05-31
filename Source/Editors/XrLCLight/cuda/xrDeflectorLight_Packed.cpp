#include "../stdafx.h"
#include "xrDeflectorLight_Packed.h"
#include <concurrent_vector.h>
#include "CUDARayCast.h"
#include "../light_point.h"
#include "../xrLC_GlobalData.h"
#include "../xrFace.h"
#include "../xrDeflector.h"
#include "../xrMU_Model_Reference.h"
#include "../XrLC/Build.h"

PackedLighting GPUTaskinSystem;
thread_local xr_vector<RayRecvestIndex>	recvest_array; 
extern void ApplyColorGPU(size_t IndexTask, base_color_c& C);
extern void ApplyColorDetailGPU(size_t IndexTask, base_color_c& C);

// Initializes
void PackedLighting::InitializeGPU()
{
	clMsg("$ InitializeGPU RayTracing");
	XRay::RayTrace::CUDA::InitializeRayTracing();
}

void PackedLighting::CleanupGPU()
{
	XRay::RayTrace::CUDA::CleanupRayTracing();
	RestartALL();
	clMsg("mem usage After GPU Cleaning:	%u mb", (u32(GetHeapMemory()) / 1024 / 1024));
}

// Deflectors
void PackedLighting::LightPointPacked_add_task(size_t IndexTask, void* Owner, Fvector& P, Fvector& N, Face* skip)
{
	// MT SAFE
	if (recvest_array.size() >= gCompilerMode.LC_CUDA_RAYS_SIZE)
		LightPointPacked_run_tasks(false);

	RayRecvestIndex task_data;
	task_data.INDEX_TASK = IndexTask;
	task_data.P = P;
	task_data.N = N;
	task_data.Owner = Owner;
	recvest_array.emplace_back(task_data);
}

void PackedLighting::LightPointPacked_run_tasks(bool need_clear)
{
	if (recvest_array.size() <= 0) return;

	// Initialize
	XRay::RayTrace::CUDA::RayTraceInitialize(current_flags, gCompilerMode.LC_CUDA_RAYS_SIZE);

	// Tasks
	for (size_t RayIndex = 0; RayIndex < recvest_array.size(); RayIndex++)
		XRay::RayTrace::CUDA::RayTraceAddRay(recvest_array[RayIndex], RayIndex);

	// Запускаем трейсинг
	XRay::RayTrace::CUDA::RayTraceRun();

	// Получаем результаты
	auto& colors = XRay::RayTrace::CUDA::RayTraceResult();
	for (auto RecvestID = 0; RecvestID < recvest_array.size(); RecvestID++)
	{
		auto& RAY_INFO = recvest_array[RecvestID];

		switch (ColorsMapType)
		{
			case eDetails:
			{
				ApplyColorDetailGPU(RAY_INFO.INDEX_TASK, colors[RecvestID]);
			}break;

			case eImplicit:
			{
				ApplyColorGPU(RAY_INFO.INDEX_TASK, colors[RecvestID]);
			}break;

			case eDeflectors:
			{
				((CDeflector*)RAY_INFO.Owner)->ApplyColor(RAY_INFO.INDEX_TASK, colors[RecvestID]);
			}break;

			case eMumodel:
			{
				((xrMU_Reference*)RAY_INFO.Owner)->colors_cuda[RAY_INFO.INDEX_TASK].add(colors[RecvestID]);;
			}break;

			case eCommon:
			{
				task_colors[RAY_INFO.INDEX_TASK].add(colors[RecvestID]);
			}break;
 		}
	}

	recvest_array.clear();
	if (need_clear)
	{
		recvest_array.shrink_to_fit();					// ThreadLocal буферы !
		XRay::RayTrace::CUDA::RayTraceCleanup();		// ThreadLocal буферы !
	}
}
