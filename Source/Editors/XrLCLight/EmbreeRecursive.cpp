#include "stdafx.h"
#include "embree4/rtcore.h"


extern RTCScene IntelScene;

void ERayQuery(RTCRayHit rayhit, float orig_range, xr_map<u32, Fvector>& m_points)
{

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.flags = RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER; // invoke filter for each geometry

	rtcIntersect1(IntelScene, &rayhit, &args);

	if (rayhit.hit.primID == RTC_INVALID_GEOMETRY_ID)
		return;

	if (rayhit.ray.tfar < rayhit.ray.tnear || rayhit.ray.tfar >= orig_range)
		return;

	Fvector intersectionPoint; intersectionPoint.set(rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z);
	intersectionPoint.mad({ rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z }, rayhit.ray.tfar);

	m_points.insert(mk_pair(u32(rayhit.hit.primID), intersectionPoint));

	rayhit.ray.org_x = intersectionPoint.x;
	rayhit.ray.org_y = intersectionPoint.y;
	rayhit.ray.org_z = intersectionPoint.z;
	rayhit.ray.tnear = EPS_L; // устанавливаем небольшое смещение относительно найденного пересечения
	rayhit.ray.tfar = orig_range - rayhit.ray.tfar; // максимальное расстояние для пересечения
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
	ERayQuery(rayhit, orig_range, m_points);

};

void RecursiveRay(Fvector pos, Fvector Dir, float Range)
{
 	xr_map<u32, Fvector> m_points = {};

	RTCRayHit rayhit;
	rayhit.ray.org_x = pos.x; // начальная точка луча
	rayhit.ray.org_y = pos.y;
	rayhit.ray.org_z = pos.z;
	rayhit.ray.dir_x = Dir.x; // направление луча
	rayhit.ray.dir_y = Dir.y;
	rayhit.ray.dir_z = Dir.z;
	rayhit.ray.tnear = 0.0f; // ближайшее расстояние для пересечения
	rayhit.ray.tfar = Range; // максимальное расстояние для пересечения

	rayhit.ray.mask = (unsigned int)(-1);
	rayhit.ray.flags = 0;

	rayhit.hit.Ng_x = 0;
	rayhit.hit.Ng_y = 0;
	rayhit.hit.Ng_z = 0;

	rayhit.hit.u = 0;
	rayhit.hit.v = 0;

	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
	ERayQuery(rayhit, Range, m_points);

 
	m_points.clear();
};