#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS_L_GPU 0.0010000f;
#define EPS_S_GPU 0.0000001f

// Hardware Vectors 
struct Hardware_Vector2
{
	float x;
	float y;

	__device__ Hardware_Vector2(float2 value)
	{
		x = value.x;
		y = value.y;
	}

	float2 __device__ getVector2()
	{
		return make_float2(x, y);
	}

	__device__ Hardware_Vector2()
	{
		x = 0;
		y = 0;
	}

	__device__ Hardware_Vector2(float InX, float InY)
	{
		x = InX;
		y = InY;
	}

	__device__ void set(float InX, float InY)
	{
		x = InX;
		y = InY;
	}

	// Math

	__device__ float DotProduct(Hardware_Vector2& Another)
	{
		return x * Another.x + y * Another.y;
	}

	__device__ float DistanceSquared(Hardware_Vector2& Another)
	{
		return (x - Another.x) * (x - Another.x) + (y - Another.y) * (y - Another.y);
	}

	__device__ Hardware_Vector2& Add(Hardware_Vector2& Another)
	{
		Hardware_Vector2 Result(0, 0);;
		Result = { x + Another.x, y + Another.y };
		x = Result.x;
		y = Result.y;
		return *this;
	}

	__device__ Hardware_Vector2& Subtract(Hardware_Vector2& Another)
	{
		Hardware_Vector2 Result(0, 0);;
		Result = { x - Another.x, y - Another.y };
		x = Result.x;
		y = Result.y;
		return *this;
	}

	__device__ Hardware_Vector2& Subtract(Hardware_Vector2& p1, Hardware_Vector2& p2)
	{
		Hardware_Vector2 Result(0, 0);;
		Result = { p1.x - p2.x, p1.y - p2.y };
		x = Result.x;
		y = Result.y;
		return *this;
	}


	__device__ void Mad_Self(Hardware_Vector2& Another, float Influence)
	{
		x += Another.x * Influence;
		y += Another.y * Influence;
	}

	// Invert
	__device__ Hardware_Vector2& Inverted()
	{
		x = -x;
		y = -y;
		return *this;
	}

	__device__ void Inverted(Hardware_Vector2 other)
	{
		x = -other.x;
		y = -other.y;
	}


	// Simular
 	__device__ bool similar(Hardware_Vector2& p, float eu, float ev) const
	{
		return abs(x - p.x) < eu && abs(y - p.y) < ev;
	}

	__device__ bool similar(const Hardware_Vector2& p, float E) const
	{
		return abs(x - p.x) < E && abs(y - p.y) < E;
	};
};

struct Hardware_Vector
{
	float x;
	float y;
	float z;

	// Cuda Convert
	__device__ Hardware_Vector(float3 value)
	{
		x = value.x;
		y = value.y;
		z = value.z;
	}

	float3 __device__ getVector3()
	{
		return make_float3(x, y, z);
	}

	// Oxygen Code
	__device__ Hardware_Vector()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	__device__ Hardware_Vector(float InX, float InY, float InZ)
	{
		x = InX;
		y = InY;
		z = InZ;
	}

	__device__ Hardware_Vector& Inverted()
	{
		x = -x;
		y = -y;
		z = -z;
		return *this;
	}

	__device__ void Inverted(Hardware_Vector other)
	{
		x = -other.x;
		y = -other.y;
		z = -other.z;
	}

	__device__ Hardware_Vector& CrossProduct(Hardware_Vector& v1, Hardware_Vector& v2)
	{
		Hardware_Vector res(0, 0, 0);
		res.x = v1.y * v2.z - v1.z * v2.y;
		res.y = v1.z * v2.x - v1.x * v2.z;
		res.z = v1.x * v2.y - v1.y * v2.x;
		return  *this;
	}

	__device__ float DotProduct(Hardware_Vector& Another)
	{
		return x * Another.x + y * Another.y + z * Another.z;
	}

	__device__ float DistanceSquared(Hardware_Vector& Another)
	{
		return
			(x - Another.x) * (x - Another.x) +
			(y - Another.y) * (y - Another.y) +
			(z - Another.z) * (z - Another.z);
	}

	__device__ Hardware_Vector& Add(Hardware_Vector& Another)
	{
		Hardware_Vector Result(0, 0, 0);;
		Result = { x + Another.x, y + Another.y, z + Another.z };
		x = Result.x;
		y = Result.y;
		z = Result.z;
		return *this;
	}

	__device__ Hardware_Vector& Add(float s)
	{
		Hardware_Vector Result(0, 0, 0);;
		Result = { x + s, y + s, z + s };
		x = Result.x;
		y = Result.y;
		z = Result.z;
		return *this;
	}

	__device__ Hardware_Vector& Subtract(Hardware_Vector& Another)
	{
		Hardware_Vector Result(0, 0, 0);;
		Result = { x - Another.x, y - Another.y, z - Another.z };
		x = Result.x;
		y = Result.y;
		z = Result.z;
		return *this;
	}

	__device__ Hardware_Vector& Subtract(Hardware_Vector& a, Hardware_Vector& b)
	{
		Hardware_Vector Result(0, 0, 0);;
		Result = { a.x - b.x, a.y - b.y, a.z - b.z };
		x = Result.x;
		y = Result.y;
		z = Result.z;
		return *this;
	}

	__device__ void Mad_Self(Hardware_Vector& Another, float Influence)
	{
		x += Another.x * Influence;
		y += Another.y * Influence;
		z += Another.z * Influence;
	}

	__device__ void MadOthers(Hardware_Vector& p, Hardware_Vector& d, float Influence)
	{
		x = p.x + d.x * Influence;
		y = p.y + d.y * Influence;
		z = p.z + d.z * Influence;
	}

	__device__ void set(float InX, float InY, float InZ)
	{
		x = InX;
		y = InY;
		z = InZ;
	}

#ifdef __CUDACC__
	__device__ void GetRangeNormalized(float sqD, float& R)
	{
		float invR = rsqrtf(sqD);
		R = sqD * invR;
		x *= invR;
		y *= invR;
		z *= invR;
	}
#endif

	__device__ void	from_bary(const Hardware_Vector& V1, const Hardware_Vector& V2, const Hardware_Vector& V3, float u, float v, float w)
	{
		x = V1.x * u + V2.x * v + V3.x * w;
		y = V1.y * u + V2.y * v + V3.y * w;
		z = V1.z * u + V2.z * v + V3.z * w;
	}

	__device__ void	from_bary(const Hardware_Vector& V1, const Hardware_Vector& V2, const Hardware_Vector& V3, const Hardware_Vector& B)
	{
		from_bary(V1, V2, V3, B.x, B.y, B.z);
	}
};

// Hardware Deflectors Stuff
struct Hardware_Color
{
	Hardware_Vector RGB;
	float hemi;
	float sun;

	__device__ Hardware_Color()
	{
		RGB.set(0,0,0);
		hemi = 0.0f;
		sun = 0.0f;
	}

	__device__ void mul(float s)
	{
		RGB.Add(s);
		hemi *= s;
		sun *= s;
	}

	__device__ void add(float s)
	{
		RGB.Add(s);
		hemi += s;
		sun += s;
	}

	__device__ void add(Hardware_Color& s)
	{
		RGB.Add(s.RGB);
		hemi += s.hemi;
		sun += s.sun;
	}
};

enum eTypeGPU
{
	eSun = 0,
	eHemi = 1,
	eRGB = 2
};

struct Hardware_Lighting
{
	uint16_t type;				// Type of light source		
	eTypeGPU light_type;		// RGB, SUN, HEMI
	float3	 diffuse;			// Diffuse color of light	
	float3	 position;			// Position in world space	
	float3	 direction;			// Direction in world space	
	float	 range;				// Cutoff range
	float	 range2;			// ^2
	float	 falloff;			// precalc to make light aqal to zero at light range
	float	 attenuation0;		// Constant attenuation		
	float	 attenuation1;		// Linear attenuation		
	float	 attenuation2;		// Quadratic attenuation	
	float	 energy;			// For radiosity ONLY
};

struct Hit
{
	float maxT;
	float minT;
};

struct Hardware_Raytask
{
	float3   Position;
	float3   Direction;
};

struct Hardware_TextureData
{
	unsigned int  width;
	unsigned int  height;
	unsigned char* pSurface; // Указатель на GPU память (Только Alpha)
};

struct Hardware_FaceData
{
	__half2		TC0[3]; // UV координаты
	unsigned short surfidx;
	bool		 bOpacue = false;
	bool		 bWater = false;
};

struct OPTICK_Params
{
	OptixTraversableHandle handle;

	unsigned char		  flags;
	Hardware_Raytask* rays;
	Hardware_Color* colors;		// Раньше rays == colors

	Hardware_Lighting* lights;
	int					  counts_lights;

	Hardware_FaceData* faces;
	int					  count_faces;

	Hardware_TextureData* textures;
	int					  count_textures;
};