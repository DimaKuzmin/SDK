//Giperion September 2016
//xrLC Redux project
//Hardware light support class

#pragma once
#include "../../xrCDB/xrCDB.h"
#include "xrRayDefinition.h"
#include "base_lighting.h"
#include "b_build_texture.h"
#include "optix\putil\Buffer.h"
#include "xrFaceDefs.h"
#include "base_color.h"

#include "DeviceBuffer.h"
#include "cuda_runtime.h"



class XRLC_LIGHT_API xrHardwareLight
{
public:
	struct MemoryCudaSTR
	{
		size_t MemoryAllocated = 0;
	};
	MemoryCudaSTR MemoryCuda;

	enum class Mode
	{
		CPU,
		OpenCL,
		CUDA
	};

	struct MemoryUse
	{
		u64 start_1;
		u64 start_2;
		u64 start_3;
		u64 start_4;
		u64 start_5;
		u64 start_6;
	};

	MemoryUse MemoryUSE;

public:

	static BOOL IsHardwareAccelerationSupported(xrHardwareLight::Mode mode);

	static xrHardwareLight& Get();

	static bool IsEnabled();

	xrHardwareLight();
	~xrHardwareLight();

	void PerformRaycast1024(xr_vector<RayRequest>& InRays, int flag, xr_vector<base_color_c>& OutHits);

	void LoadLevel(CDB::MODEL* RaycastModel, base_lighting& Ligtings, xr_vector<b_BuildTexture>& Textures);

	void PerformRaycast(xr_vector<RayRequest>& InRays, int flag, xr_vector<base_color_c>& OutHits);

	//void CalculateLightmap(int DeflectorID, struct lm_layer& LightMapRef);

	void PerformAdaptiveHT();

	float GetEnergyFromSelectedLight(xr_vector<int>& RGBLightIndexes, xr_vector<int>& SunLightIndexes, xr_vector<int>& HemiLightIndexes);

	xr_vector<Fvector> GetDebugPCHitData();
	xr_vector<Fvector> GetDebugGPUHitData();

	//here goes special functions for new batching concept

	void LoadRaycastModel(CDB::MODEL* RaycastModel, xr_vector<RayRequest>& InRays);
 
	void TriFindPos(xr_vector<RayRequest>& InRays);

private:

	Mode mode;
	void GetLevelIndices(vecVertex& InLevelVertices, vecFace& InLevelFaces, xr_vector <PolyIndexes>& OutLevelIndices, xr_vector<HardwareVector>& OutLevelVertexes);

	__forceinline void CheckCudaError(cudaError_t ErrorCode);

	size_t GetDeviceFreeMem();
	size_t GetMemoryRequiredForLoadLevel(CDB::MODEL* RaycastModel, base_lighting& Lightings, xr_vector<b_BuildTexture>& Textures);

	const char* ToString_Mode(Mode mode);

	//Master struct
	DeviceBuffer<xrHardwareLCGlobalData>* GlobalData;

	//Light data
	DeviceBuffer<LightSizeInfo>* LightSizeBuffer;
	DeviceBuffer<R_Light>* LightBuffer;

	//Level geometry data
	DeviceBuffer<PolyIndexes>* TrisBuffer;
	DeviceBuffer<Fvector>* VertBuffer;
	DeviceBuffer<HardwareVector>* VertNormalBuffer;

	//Textures
	DeviceBuffer<xrHardwareTexture>* TextureBuffer;
	xr_vector< DeviceBuffer < char >* > TexturesData;

	// Memory Statistics
	size_t DeviceMemoryForLevel;

	static bool _IsEnabled;

};
