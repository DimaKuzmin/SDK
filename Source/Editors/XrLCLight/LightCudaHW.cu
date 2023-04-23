#ifdef __INTELLISENSE__
#define __global__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "../../xrCore/_types.h"

__device__ float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


__device__ float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


__device__ float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__ Triangle& getTriangle(int id, Triangle* triangles) {
	return triangles[id];
}


__device__ BVHNode& getNode(int id, BVHNode* nodes) {
	return nodes[id];
}


__device__ bool intersectTriangle(const float3& orig, const float3& dir, const Triangle& tri, float& t, float& u, float& v) {
	const float3 pvec = cross(dir, tri.e2);
	const float det = dot(tri.e1, pvec);
	if (det < 1e-7f) return false;
	const float3 tvec = orig - tri.v0;
	u = dot(tvec, pvec);
	if (u < 0.0f || u > det) return false;
	const float3 qvec = cross(tvec, tri.e1);
	v = dot(dir, qvec);
	if (v < 0.0f || u + v > det) return false;
	t = dot(tri.e2, qvec);
	const float invDet = 1.0f / det;
	t *= invDet;
	u *= invDet;
	v *= invDet;
	return true;
}


__global__ void intersectBVH(const BVHNode* dnodes, const Triangle* dtriangles, const float3* dorig, const float3* ddir, int* dresults, int* dcount) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= dcount) return;
	const BVHNode nodes = dnodes;
	const Triangle* triangles = dtriangles;
	const float3 orig = dorig[tid];
	const float3 dir = ddir[tid];
	int stack[64], sp = 0, id;
	stack[sp++] = 0;
	while (sp > 0) {
		id = stack[--sp];
		const BVHNode& node = getNode(id, nodes);
		if (dot(orig - node.max, dir) > 0.0f) continue;
		if (dot(orig - node.min, dir) < 0.0f) {
			if (node.left >= 0) stack[sp++] = node.left;
			if (node.right >= 0) stack[sp++] = node.right;
			continue;
		}
		if (node.left < 0) {
			const Triangle& tri = getTriangle(-node.left - 1, triangles);
			float t, u, v;
			if (intersectTriangle(orig, dir, tri, t, u, v)) {
				dresults[tid] = tri.id;
				atomicAdd(d_count, 1);
			}
		}
		else {
			stack[sp++] = node.left;
			stack[sp++] = node.right;
		}
	}
}


int main() {
	Triangle* htriangles = new Triangle[MAXTRIANGLES];
	BVHNode* hnodes = new BVHNode[MAXTRIANGLES * 2 - 1];
	float3* horig = new float3[MAXTRIANGLES];
	float3* hdir = new float3[MAXTRIANGLES];
	int* hresults = new int[MAXTRIANGLES];
	int* dresults, * dcount;
	cudaMalloc(&dresults, sizeof(int) * MAXTRIANGLES);
	cudaMalloc(&dcount, sizeof(int));
	int count = 0;
	// fill htriangles and hnodes with BVH hierarchy data
	// fill horig and hdir with ray origin and direction data
	cudaMemcpy(dresults, h, results, sizeof(int) * MAXTRIANGLES, cudaMemcpyHostToDevice);
	cudaMemcpy(dcount, &count, sizeof(int), cudaMemcpyHostToDevice);
	const int threadsPerBlock = 256;
	const int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
	intersectBVH<<>>(dnodes, dtriangles, dorig, ddir, dresults, dcount);
	cudaMemcpy(hresults, dresults, sizeof(int) * MAXTRIANGLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(&count, dcount, sizeof(int), cudaMemcpyDeviceToHost);
	// process results in hresults
	delete[] htriangles;
	delete[] hnodes;
	delete[] horig;
	delete[] hdir;
	delete[] hresults;
	cudaFree(dresults);
	cudaFree(d_count);
	return 0;
}