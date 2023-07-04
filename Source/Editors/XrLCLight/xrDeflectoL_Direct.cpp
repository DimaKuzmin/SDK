#include "stdafx.h"
//#include "build.h"
//#include "std_classes.h"
#include "xrThread.h"
#include "xrdeflector.h"
#include "xrlc_globaldata.h"
#include "light_point.h"
#include "xrface.h"
#include "net_task.h"

#include "xrHardwareLight.h"
#include "EmbreeDataStorage.h"

#include "xrLight_Implicit.h"

extern void Jitter_Select	(Fvector2* &Jitter, u32& Jcount);

xr_map<CDeflector*, xr_vector<LightpointRequest>> deflectors_edges;
xr_map<CDeflector*, xr_vector<LightpointRequest>> deflectors_recvests;
 
extern bool use_intel = false;


void CDeflector::L_Direct_Edge (int th, CDB::COLLIDER* DB, base_lighting* LightsSelected, Fvector2& p1, Fvector2& p2, Fvector& v1, Fvector& v2, Fvector& N, float texel_size, Face* skip)
{
	Fvector		vdir;
	vdir.sub	(v2,v1);
	
	lm_layer&	lm	= layer;
	
	Fvector2		size; 
	size.x			= p2.x-p1.x;
	size.y			= p2.y-p1.y;
	int	du			= iCeil(_abs(size.x)/texel_size);
	int	dv			= iCeil(_abs(size.y)/texel_size);
	int steps		= _max(du,dv);
	if (steps<=0)	return;
	
	for (int I=0; I<=steps; I++)
	{
		float	time = float(I)/float(steps);
		Fvector2	uv;
		uv.x	= size.x*time+p1.x;
		uv.y	= size.y*time+p1.y;
		int	_x  = iFloor(uv.x*float(lm.width)); 
		int _y	= iFloor(uv.y*float(lm.height));
		
		if ((_x<0)||(_x>=(int)lm.width))	continue;
		if ((_y<0)||(_y>=(int)lm.height))	continue;
		
		if (lm.marker[_y*lm.width+_x])		continue;
		
		// ok - perform lighting
		base_color_c	C;
		Fvector			P;	P.mad(v1,vdir,time);
		VERIFY(inlc_global_data());
		VERIFY(inlc_global_data()->RCAST_Model());

		if (!xrHardwareLight::IsEnabled())
		{
			if (use_intel)
			{
				u32 flags = (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | LP_DEFAULT;

				if (0 == (flags & LP_dont_sun))
					RaysToSUNLight_Deflector(th, P, N, C, *LightsSelected, skip);
				if (0 == (flags & LP_dont_hemi))
					RaysToHemiLight_Deflector(th, P, N, C, *LightsSelected, skip);
				if (0 == (flags & LP_dont_rgb))
					RaysToRGBLight_Deflector(th, P, N, C, *LightsSelected, skip);
			}
			else
				LightPoint(DB, inlc_global_data()->RCAST_Model(), C, P, N, *LightsSelected, (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | LP_DEFAULT, skip, 1024); //.
			C.mul(.5f);
			lm.surface[_y * lm.width + _x]._set(C);
			lm.marker[_y * lm.width + _x] = 255;
		}
		else
		{
			deflectors_edges[this].push_back(LightpointRequest(_x, _y, P, N, skip));
			lm.marker[_y * lm.width + _x] = 255;
		}

	}
}

bool sort_rays(RayOptimizedTyped a, RayOptimizedTyped b)
{
	bool cmp1 = a.dir.x < b.dir.x&& a.dir.y < b.dir.y&& a.dir.z < b.dir.z;;
	//bool cmp2 = a.pos.x < b.pos.x&& a.pos.y < b.pos.y&& a.pos.z < b.pos.z;

	return cmp1 ; //&& cmp2
}





int id = 0;

#include <tbb/parallel_for_each.h>
	 
void CDeflector::L_Direct	(int th, CDB::COLLIDER* DB, base_lighting* LightsSelected, HASH& H)
{
	id++;
	R_ASSERT	(DB);
	R_ASSERT	(LightsSelected);

	lm_layer&	lm = layer;

	// Setup variables
	Fvector2	dim,half;
	dim.set		(float(lm.width),float(lm.height));
	half.set	(.5f/dim.x,.5f/dim.y);
	
	// Jitter data
	Fvector2	JS;
	JS.set		(.4999f/dim.x, .4999f/dim.y);
	
	u32			Jcount;
	Fvector2*	Jitter;
	Jitter_Select(Jitter, Jcount);
	
	// Lighting itself
	DB->ray_options	(0);

	//xr_vector<RayOptimizedTyped>  rays;
	
	for (u32 V=0; V<lm.height; V++)
	{
		if(_net_session && !_net_session->test_connection())
			 return;
		for (u32 U=0; U<lm.width; U++)	
		{
#ifdef NET_CMP
			if(V*lm.width+U!=8335)
				continue;
#endif
			u32				Fcount	= 0;
			base_color_c	C;
			
			try 
			{
				for (u32 J=0; J<Jcount; J++) 
				{
					// LUMEL space
					Fvector2 P;
					P.x = float(U)/dim.x + half.x + Jitter[J].x * JS.x;
					P.y = float(V)/dim.y + half.y + Jitter[J].y * JS.y;
					
					xr_vector<UVtri*>&	space	= H.query(P.x,P.y);
					
					// World space
					Fvector		wP,wN,B;
					for (UVtri** it=&*space.begin(); it!=&*space.end(); it++)
					{
						if ((*it)->isInside(P,B) )
						{
							// We found triangle and have barycentric coords
							Face	*F	= (*it)->owner;
							Vertex	*V1 = F->v[0];
							Vertex	*V2 = F->v[1];
							Vertex	*V3 = F->v[2];
							wP.from_bary(V1->P,V2->P,V3->P,B);
//. не нужно использовать	if (F->Shader().flags.bLIGHT_Sharp)	{ wN.set(F->N); }
//							else								
							{ 
								wN.from_bary(V1->N,V2->N,V3->N,B);	exact_normalize	(wN); 
								wN.add		(F->N);					exact_normalize	(wN);
							}
							
							if (!xrHardwareLight::IsEnabled())
							{
								try
								{
									VERIFY(inlc_global_data());
									VERIFY(inlc_global_data()->RCAST_Model());
									if (use_intel)
									{
										// LP_UseFaceDisable НЕ ИСПОЛЬЗУЕТСЯ НИГДЕ !!!
										u32 flags = (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0);

										if (0 == (flags & LP_dont_sun))
											RaysToSUNLight_Deflector(th, wP, wN, C, *LightsSelected, F);
										if (0 == (flags & LP_dont_hemi))
											RaysToHemiLight_Deflector(th, wP, wN, C, *LightsSelected, F);
										if (0 == (flags & LP_dont_rgb))
											RaysToRGBLight_Deflector(th, wP, wN, C, *LightsSelected, F);
 									}
									else 
										LightPoint(DB, inlc_global_data()->RCAST_Model(), C, wP, wN, *LightsSelected, (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | LP_UseFaceDisable, F, 1024); //.
									
									Fcount += 1;
								}
								catch (...)
								{
									clMsg("* ERROR (CDB). Recovered. ");
								}
							}
							else
							{
								//LightPoint(DB, inlc_global_data()->RCAST_Model(), C, wP, wN, *LightsSelected, (inlc_global_data()->b_norgb() ? LP_dont_rgb : 0) | (inlc_global_data()->b_nosun() ? LP_dont_sun : 0) | (inlc_global_data()->b_nohemi() ? LP_dont_hemi : 0) | LP_UseFaceDisable, F); //.
								Fcount += 1;
								//lm.SurfaceLightRequests.push_back(LightpointRequest(U,V, wP, wN, F));
								deflectors_recvests[this].push_back(LightpointRequest(U, V, wP, wN, F));
								lm.marker[V * lm.width + U] = 255;
							}
							

							break;
						}
					}
				} 
			} 
			catch (...) 
			{
				clMsg("* ERROR (Light). Recovered. ");
			}

			if (!xrHardwareLight::IsEnabled() && !use_intel)
			{
				if (Fcount)
				{
					C.scale(Fcount);
					C.mul(.5f);
					lm.surface[V * lm.width + U]._set(C);
					lm.marker[V * lm.width + U] = 255;
				}
				else
				{
					lm.surface[V * lm.width + U]._set(C);	// 0-0-0-0-0
					lm.marker[V * lm.width + U] = 0;
				}
			}
		}
	}										     
 
	// *** Render Edges
	float texel_size = (1.f/float(_max(lm.width,lm.height)))/8.f;
	for (u32 t=0; t<UVpolys.size(); t++)
	{
		UVtri&		T	= UVpolys[t];
		Face*		F	= T.owner;
		R_ASSERT	(F);
		try 
		{
			L_Direct_Edge	(th, DB,LightsSelected, T.uv[0], T.uv[1], F->v[0]->P, F->v[1]->P, F->N, texel_size,F);
			L_Direct_Edge	(th, DB,LightsSelected, T.uv[1], T.uv[2], F->v[1]->P, F->v[2]->P, F->N, texel_size,F);
			L_Direct_Edge	(th, DB,LightsSelected, T.uv[2], T.uv[0], F->v[2]->P, F->v[0]->P, F->N, texel_size,F);
		}
		catch (...)
		{
			clMsg("* ERROR (Edge). Recovered. ");
		}
	}
 
}	  













#include "cl_intersect.h"
#include "R_light.h"

void RaysToHemiLights(HardwareVector& P, HardwareVector& N, base_lighting& lights, xr_vector<Ray>& rays)
{
	Fvector		Ldir, Pnew;
	Pnew.mad(P, N, 0.01f);
	R_Light* L = &*lights.hemi.begin(), * E = &*lights.hemi.end();

	for (; L != E; L++)
	{
		if (L->type == LT_DIRECT)
		{
			// Cos
			Ldir.invert(L->direction);
			float D = Ldir.dotproduct(N);
			if (D <= 0) continue;

			// Trace Light
			Fvector		PMoved;
			PMoved.mad(Pnew, Ldir, 0.001f);

			Ray r;
			r.Origin = PMoved;
			r.Direction = Ldir;
			r.tmax = 1000.f;
			r.tmin = 0.f;

			rays.push_back(r);
		}
		else
		{
			HardwareVector pos = L->position;

			// Distance
			float sqD = P.DistanceSquared(pos);
			if (sqD > L->range2) continue;

			// Dir
			Ldir.sub(L->position, P);
			Ldir.normalize_safe();
			float D = Ldir.dotproduct(N);
			if (D <= 0) continue;

			// Trace Light
 			Ray r;
			r.Origin = Pnew;
			r.Direction = Ldir;
			r.tmax = 1000.f;
			r.tmin = 0.f;
			rays.push_back(r);

		}

	}
}



#include "optix/optix_prime.h"
#include "optix/optix_primepp.h"
#include "cuda_runtime.h"

//level buffers
optix::prime::Context PrimeContext_NEW;
optix::prime::Model LevelModel_OPTIX;

#define CHK_CUDA( code )                                                       \
{                                                                              \
  cudaError_t err__ = code;                                                    \
  if( err__ != cudaSuccess )                                                   \
  {                                                                            \
    Msg("Error at (%s), line: (%d), code: (%d)",  __FILE__ ,  __LINE__ , code);              \
    exit(1);                                                                   \
  }                                                                            \
}

void InitWorldModelOptix(CDB::MODEL* RaycastModel)
{
	PrimeContext_NEW = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);

	DeviceBuffer<HardwareVector> CDBVertexesBuffer;
	DeviceBuffer<PolyIndexes> CDBTrisIndexBuffer;

	DeviceBuffer<TrisAdditionInfo> CDBTrisAdditionBuffer;

	xr_vector<PolyIndexes> OptimizedMeshTris;
	OptimizedMeshTris.reserve(RaycastModel->get_tris_count());

	xr_vector<TrisAdditionInfo> OptimizedTrisAdditionInfo;
	OptimizedTrisAdditionInfo.reserve(RaycastModel->get_tris_count());

	// ����� ���� ������������� ������
	for (int i = 0; i < RaycastModel->get_tris_count(); i++)
	{
		CDB::TRI Tris = RaycastModel->get_tris()[i];
		PolyIndexes indx{ (u32)Tris.verts[0], (u32)Tris.verts[1], (u32)Tris.verts[2] };

		OptimizedMeshTris.push_back(indx);

		TrisAdditionInfo AdditionInfo;

		base_Face& FaceRef = *(base_Face*)Tris.pointer;

		const Shader_xrLC& TrisShader = FaceRef.Shader();

		AdditionInfo.CastShadow = !!TrisShader.flags.bLIGHT_CastShadow;

		Fvector2* pCDBTexCoord = FaceRef.getTC0();
		AdditionInfo.TexCoords = *pCDBTexCoord;

		b_material& TrisMaterial = inlc_global_data()->materials()[FaceRef.dwMaterial];
		AdditionInfo.TextureID = TrisMaterial.surfidx;

		//#WARNING: :(
		AdditionInfo.FaceID = (u64)&FaceRef;
		OptimizedTrisAdditionInfo.push_back(AdditionInfo);
	}

	CDBTrisIndexBuffer.alloc(OptimizedMeshTris.size(), DeviceBufferType::CUDA, 1);
	CDBTrisIndexBuffer.copyToBuffer(OptimizedMeshTris.data(), OptimizedMeshTris.size());

	CDBTrisAdditionBuffer.alloc(OptimizedTrisAdditionInfo.size(), DeviceBufferType::CUDA, 1);
	CDBTrisAdditionBuffer.copyToBuffer(OptimizedTrisAdditionInfo.data(), OptimizedTrisAdditionInfo.size());

	CDBVertexesBuffer.alloc(RaycastModel->get_verts_count(), DeviceBufferType::CUDA, 1);
	CDBVertexesBuffer.copyToBuffer(reinterpret_cast<HardwareVector*>(RaycastModel->get_verts()), RaycastModel->get_verts_count());

	Msg("SizeTRI: %d, Aditinal: %d, SizeVert: %d", CDBTrisIndexBuffer.count(), CDBTrisAdditionBuffer.count(), CDBVertexesBuffer.count());

	// ������� ������ � OptiX Prime

	optix::prime::BufferDesc LevelIndixes_CDB;
	optix::prime::BufferDesc LevelVertexes_CDB;

	LevelIndixes_CDB = PrimeContext_NEW->createBufferDesc(RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBTrisIndexBuffer.ptr());
	LevelIndixes_CDB->setRange(0, CDBTrisIndexBuffer.count());

	LevelVertexes_CDB = PrimeContext_NEW->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_CUDA_LINEAR, CDBVertexesBuffer.ptr());
	LevelVertexes_CDB->setRange(0, CDBVertexesBuffer.count());


	LevelModel_OPTIX = PrimeContext_NEW->createModel();
	// LevelModel_OPTIX ������� � ������� ������
	LevelModel_OPTIX->setTriangles(LevelIndixes_CDB, LevelVertexes_CDB);
	// ��������� ������
	LevelModel_OPTIX->update(0);



}


void ExecuteOptix(xr_vector<Ray>& rays, xr_vector<Hit>& hits)
{
	if (rays.size() == 0)
	{
		Msg("!!!! buffer size: Hits count: %d, Rays count: %d", hits.size(), rays.size());

		return;
	}

	// ������� ������ � OptiX Prime
	optix::prime::Query query = LevelModel_OPTIX->createQuery(RTP_QUERY_TYPE_CLOSEST);

	// ������� ����� ��� ����������� 	 
	DeviceBuffer<Ray> OptimizedRaysVec(rays.size());
	OptimizedRaysVec.copyToBuffer(rays.data(), rays.size());

	DeviceBuffer<Hit> OptimizedHitsVec(rays.size());
	CHK_CUDA(cudaMemset(OptimizedHitsVec.ptr(), 0, OptimizedHitsVec.count() * sizeof(Hit)));

	optix::prime::BufferDesc OptimizedRaysDescBuffer = PrimeContext_NEW->createBufferDesc((RTPbufferformat)Ray::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedRaysVec.ptr());
	OptimizedRaysDescBuffer->setRange(0, OptimizedRaysVec.count());

	optix::prime::BufferDesc OptmizedHitDescBuffer = PrimeContext_NEW->createBufferDesc((RTPbufferformat)Hit::format, RTPbuffertype::RTP_BUFFER_TYPE_CUDA_LINEAR, OptimizedHitsVec.ptr());
	OptmizedHitDescBuffer->setRange(0, OptimizedHitsVec.count());

	Msg("Size: rays %d, size: hits: %d", OptimizedRaysVec.count(), OptimizedHitsVec.count());

	query->setRays(OptimizedRaysDescBuffer);
	query->setHits(OptmizedHitDescBuffer);

	CTimer t;
	t.Start();
	query->execute(0);

	Msg("GetElapsed: %d", t.GetElapsed_ms());

	/*
	// ������������ ���������� �������
	if (result.isValid())
	{
		float distance = result.distance;
		int hitPrimitiveIndex = result.hitIndex;
		int hitInstaceIndex = result.instanceIndex;
		float2 barycentrics = result.barycentrics;
		float3 hitPoint = result.worldPosition;
		float3 hitNormal = result.worldNormal;
		// ...
	}
	else 
	{
 	}
	*/

}

void GPU_Calculation()
{
	xrHardwareLight& HardwareCalculator = xrHardwareLight::Get();

	xr_vector<base_color_c> FinalColors;
	xr_vector<RayRequest> RayRequests;

	 
	for (auto defl : deflectors_recvests)
	for (auto ray : defl.second)
		RayRequests.push_back(RayRequest{ ray.Position, ray.Normal, ray.FaceToSkip});
	 

	Msg("DSize: %d, RaySize: %d", deflectors_recvests.size(), RayRequests.size());
 
	xr_vector<Ray> rays;
	InitWorldModelOptix(lc_global_data()->RCAST_Model());

	int hits_total = 0;

	int i = 0;	  
	u64 total_Rays = 0;

	CTimer buffer; buffer.Start();

	for (auto ray : RayRequests)
	{
		RaysToHemiLights(ray.Position, ray.Normal, inlc_global_data()->L_static(), rays);
		/*
		if (i % 800000 == 0)
		{
			Msg("ID: %d/%d, size: %d", i, RayRequests.size(), rays.size());
			log_vminfo();
			xr_vector<Hit> hits;
			hits.reserve(rays.size());
			ExecuteOptix(rays, hits);
			rays.clear();	  
			hits_total += hits.size();
		}
		*/
		 
		if (i % 800000 == 0)
		{
			Msg("ID: %d, MS: %d, size: %d, total_calculated: %llu", i, buffer.GetElapsed_ms(), RayRequests.size(), total_Rays);
			total_Rays += rays.size();
			rays.clear();
		}

		i++;
	}

	Msg("!!! ID: %d, MS: %d, total: %llu", i, buffer.GetElapsed_ms(), total_Rays);

	rays.clear();
	
}

   