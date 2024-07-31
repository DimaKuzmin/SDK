#include "stdafx.h"
#include "xrdeflector.h" 
#include "EmbreeDataStorage.h"
#include "base_lighting.h"

void RayTraceEmbree8Preocess(PackedBuffer* buffer, ELightType type_lightpoint, ELights type_LIGHTs);

void LightPointPacked(PackedBufferTOProcess* buffer_to_work, base_lighting& lights, u32 flags)
{
	Fvector		Ldir;

	/*
	if (0 == (buffer->flags[0] & LP_dont_rgb))
	{
		R_Light* L = &*lights.rgb.begin(), * E = &*lights.rgb.end();
		for (; L != E; L++)
		{
			buffer->light = L;

			switch (L->type)
			{
				case LT_DIRECT:
				{
					int valid[8];
					bool any = false;

					for (auto i = 0; i < 8; i++)
					{
						Pnew.mad(buffer->position[i], buffer->direction[i], 0.01f);

						// Cos
						Ldir.invert(L->direction);
						float D = Ldir.dotproduct(buffer->direction[i]);

						valid[i] = false;
						buffer->Dist2Light[i] = D;

						if (D <= 0)
							continue;

						valid[i] = true;
						any = true;
					}

					if (any)
					{
						// Trace Light
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}
				}
				break;
				case LT_POINT:
				{
					int valid[8];
					bool any = false;

					for (auto i = 0; i < 8; i++)
					{
						Pnew.mad(buffer->position[i], buffer->direction[i], 0.01f);

						// Distance
						float sqD = buffer->position[i].distance_to_sqr(L->position);
						valid[i] = false;
						if (sqD > L->range2)
							continue;

						// Dir
						Ldir.sub(L->position, buffer->position[i]);
						Ldir.normalize_safe();
						float D = Ldir.dotproduct(buffer->direction[i]);
						if (D <= 0)
							continue;

						valid[i] = true;
						any = true;
						buffer->Dist2Light[i] = D;

					}

					if (any)
					{
						// Trace Light
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}

				}
				break;
				case LT_SECONDARY:
				{
					int valid[8];
					bool any = false;

					for (auto i = 0; i < 8; i++)
					{
						// Distance
						valid[i] = false;

						float sqD = buffer->position[i].distance_to_sqr(L->position);
						if (sqD > L->range2) continue;

						// Dir
						Ldir.sub(L->position, buffer->position[i]);
						Ldir.normalize_safe();
						float	D = Ldir.dotproduct(buffer->direction[i]);
						if (D <= 0) continue;

						D *= -Ldir.dotproduct(L->direction);
						if (D <= 0) continue;

						valid[i] = false;
						buffer->Dist2Light[i] = D;
						any = true;
					}

					if (any)
					{
						RayTraceEmbree8Preocess(buffer, valid, L->type);
					}
				}
				break;
			}
		}
	}
	*/

	R_Light* L = &*lights.hemi.begin(), * E = &*lights.hemi.end();

	for (; L != E; L++)
	{
		PackedBuffer buffer;
		buffer.MDL = buffer_to_work->MDL;
		buffer.light = L;
		buffer.flags = flags;



		if (L->type == LT_DIRECT)
		{
			bool any = false;

			for (int i = 0; i < 8; i++)
			{
				// Cos
				Ldir.invert(L->direction);
				float D = Ldir.dotproduct(buffer_to_work->direction[i]);
				buffer.valid[i] = -1;

				if (D <= 0)
					continue;

				any = true;
				buffer.valid[i] = 0;

				// Trace Light
				Fvector	PMoved, PNew;
				PNew.set(buffer_to_work->position[i]); PNew.add(0.01f);
				PMoved.mad(PNew, Ldir, 0.001f);

				buffer.pos[i].set(PMoved);
				buffer.dir[i].set(Ldir);
				buffer.tmax[i] = 1000.f;
				buffer.skip[i] = buffer_to_work->skip[i];
			}

			if (any)
			{
				RayTraceEmbree8Preocess(&buffer, ELightType::LT_Direct, ELights::Hemi);

				for (auto i = 0; i < 8; i++)
				{
					buffer_to_work->color[i].add(buffer.Color[i]);
				}
			}

		}
		else
		{
			bool any = false;

			for (int i = 0; i < 8; i++)
			{
				buffer.valid[i] = -1;
				// Distance
				float sqD = buffer_to_work->position[i].distance_to_sqr(L->position);
				if (sqD > L->range2) continue;

				// Dir
				Ldir.sub(L->position, buffer_to_work->position[i]);
				Ldir.normalize_safe();
				float D = Ldir.dotproduct(buffer_to_work->direction[i]);
				if (D <= 0) continue;


				any = true;
				buffer.valid[i] = 0;

				// Trace Light
				float R = _sqrt(sqD);

				Fvector	PNew;
				PNew.set(buffer_to_work->position[i]); PNew.add(0.01f);

				buffer.pos[i].set(PNew);
				buffer.dir[i].set(Ldir);
				buffer.tmax[i] = R;
				buffer.skip[i] = buffer_to_work->skip[i];

			}

			if (any)
			{
				RayTraceEmbree8Preocess(&buffer, ELightType::LT_Point, ELights::Hemi);

				for (auto i = 0; i < 8; i++)
				{
					buffer_to_work->color[i].add(buffer.Color[i]);
				}
			}

		}

	}

}
