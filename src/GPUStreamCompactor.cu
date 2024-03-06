// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM
#include <cuda_runtime_api.h>
#include <iostream>

#include "DataTypes.cuh"
#include "GPUStreamCompactor.cuh"
#include "Utility.cuh"


struct GPUTester0
{
	__device__ bool operator()(const TestType& t, uint32_t index)
	{
		return test_hash0(t, index);
	}
};
struct GPUTester1
{
	__device__ bool operator()(const TestType& t, uint32_t index)
	{
		return test_hash1(t, index);
	}
};
struct GPUTester2
{
	__device__ bool operator()(const TestType& t, uint32_t index)
	{
		return test_hash2(t, index);
	}
};

uint32_t RunGPUTest(const TestType* input, TestType* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool in_place, bool order_preserving, int test)
{
	if(test == 0)
		return GpuStreamCompactor::run<TestType>(input, output, element_count, temp_memory, temp_size, in_place, order_preserving, GPUTester0());
	if (test == 1)
		return GpuStreamCompactor::run<TestType>(input, output, element_count, temp_memory, temp_size, in_place, order_preserving, GPUTester1());
	if (test == 2)
		return GpuStreamCompactor::run<TestType>(input, output, element_count, temp_memory, temp_size, in_place, order_preserving, GPUTester2());
	return 0xFFFFFFFF;
}

// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM