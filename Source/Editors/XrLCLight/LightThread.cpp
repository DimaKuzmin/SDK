#include "stdafx.h"

#include "LightThread.h"

#include "global_calculation_data.h"
#include "mutex"

std::atomic<int> atomic = 0;
u32 MAX_SIZE = 0;
u32 MIN_SIZE = 0;

void LightThread ::Execute()
{
	CDB::COLLIDER		DB;
	DB.ray_options		( CDB::OPT_CULL	);
	DB.box_options		( CDB::OPT_FULL_TEST );
	base_lighting		Selected;

	MAX_SIZE = gl_data.slots_data.size_z();

	for (;;)
	{
		u32 Z = atomic.load();
		atomic.fetch_add(1);
		if (Z < MAX_SIZE)
		{
			// if (Z % 32 == 0)
				Status("Z: %u | X: %u", Z, gl_data.slots_data.size_z());

			for (u32 X = 0; X < gl_data.slots_data.size_x(); X++)
			{
				DetailSlot& DS = gl_data.slots_data.get_slot(X, Z);
				if (!detail_slot_process(X, Z, DS))
					continue;
				if (!detail_slot_calculate(X, Z, DS, box_result, DB, Selected))
					continue;

				gl_data.slots_data.set_slot_calculated(X, Z);
				thProgress = float(MIN_SIZE) / float(MAX_SIZE);
				thPerformance = float(double(t_count) / double(t_time * CPU::clk_to_seconds)) / 1000.f;
			}
		}
		else
			break;
	}
}