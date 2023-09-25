//Giperion August 2018
//[EUREKA] 3.6.1
// xrLC Redux, X-Ray Oxygen
// Class modified from Nvidia Buffer<T> (OptiX lib)

#pragma once

#include <vector>
#include "xrHardwareLight.h"

enum class DeviceBufferType
{
	CUDA
};
//------------------------------------------------------------------------------
//
// A simple abstraction for memory to be passed into Prime via BufferDescs
//




template<typename T>
class DeviceBuffer
{
public:
	template<class T>
	void SafeCudaMemalloc(T** ptr, size_t size_in_bites)
	{
		if (size_in_bites > 1024 * 1024 * 1024)
		{
			//Msg("MEMSET: %d MB, buffer: %s", size_in_bites / 1024 / 1024, name);
		}

		CHK_CUDA(cudaMalloc(ptr, size_in_bites));
	}

	DeviceBuffer(size_t count = 0, DeviceBufferType type = DeviceBufferType::CUDA, bool VRAM = 1, LPCSTR l = 0) : m_ptr(0), name(l), mem_type(VRAM)
	{
 		alloc(count, type, VRAM);
 	}

	// Allocate without changing type
	void alloc(size_t count)
	{
		alloc(count, m_type);
	}


	void alloc(size_t count, DeviceBufferType type, bool VRAM)
	{
		if (m_ptr)
			free();
		mem_type = VRAM;

		m_type = type;
		m_count = count;


		if (m_count > 0)
		{
			if (m_type == DeviceBufferType::CUDA)
			{
				CHK_CUDA(cudaGetDevice(&m_device));

				if (VRAM)
				{
					SafeCudaMemalloc(&m_ptr, sizeInBytes());
				}
				else
				{
					//Msg("use sys mem, buffer name : %s", name);
					CHK_CUDA(cudaMallocHost(&m_ptr, sizeInBytes(), 0));
				}
				 
			}
			else
			{
				FATAL("Undefined Buffer::Type for Buffer");
			}
		}
	}

	void free()
	{
		if (m_type == DeviceBufferType::CUDA)
		{
			int oldDevice;
			CHK_CUDA(cudaGetDevice(&oldDevice));
			CHK_CUDA(cudaSetDevice(m_device));
			if (mem_type == 1)
			{
				CHK_CUDA(cudaFree(m_ptr));
			}
			else
			{
				CHK_CUDA(cudaFreeHost(m_ptr));
			}
			 
			CHK_CUDA(cudaSetDevice(oldDevice));
		}		 
		else
		{
			FATAL("Undefined Buffer::Type for Buffer");
		}

		xrHardwareLight::Get().MemoryCuda.MemoryAllocated -= m_count;

		m_ptr = 0;
		m_count = 0;
	}

	~DeviceBuffer()
	{
		free();
	}

	size_t count()       const { return m_count; }
	size_t sizeInBytes() const { return m_count * sizeof(T); }
	const T* ptr()       const
	{
		return m_ptr;
	}
	T* ptr()
	{
		return m_ptr;
	}
	DeviceBufferType type() const { return m_type; }

	const T* hostPtr()
	{
		if (m_type == DeviceBufferType::CUDA)
		{
			m_tempHost.resize(m_count);
			CHK_CUDA(cudaMemcpy(&m_tempHost[0], m_ptr, sizeInBytes(), cudaMemcpyDeviceToHost));
			return &m_tempHost[0];
		}
		else
		{
			FATAL("Undefined Buffer::Type for Buffer");
		}

		return nullptr;
	}
	
	std::vector<T> hostPtrVec()
	{
		m_tempHost.resize(m_count);
		CHK_CUDA(cudaMemcpy(m_tempHost.data(), m_ptr, sizeInBytes() * sizeof(T), cudaMemcpyDeviceToHost));
		return m_tempHost;
	}

	// count == elements
	void copyToBuffer(T* InData, size_t count, size_t offset = 0)
	{
		if (m_type == DeviceBufferType::CUDA)
		{
			cudaError_t ErrCode = cudaMemcpy(m_ptr + offset, InData, count * sizeof(T), cudaMemcpyHostToDevice);
			R_ASSERT(ErrCode == cudaSuccess);
		}
		else
		{
			FATAL("Undefined Buffer::Type for Buffer");
		}
	}

protected:
	DeviceBufferType m_type;
	T* m_ptr;
	int m_device;
	size_t m_count;
	std::vector<T> m_tempHost;
	bool mem_type;

	LPCSTR name;
 
private:
	DeviceBuffer<T>(const DeviceBuffer<T>&);            // forbidden
	DeviceBuffer<T>& operator=(const DeviceBuffer<T>&); // forbidden
};

