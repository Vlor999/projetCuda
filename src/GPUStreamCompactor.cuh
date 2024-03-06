#include "GPUStreamCompactor.h"
#include <cuda_runtime_api.h>
#include <iostream>

#include "DataTypes.cuh"
#include "Utility.cuh"

#include <cub/cub.cuh>


// example kernel that evaluates whether elements should be kept
// note that the evaluation functor is passed per value here
template<typename T, typename F>
__global__ void writeState(const T* input, uint32_t* state, uint32_t num_elements, F f)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= num_elements)
		return;

	bool keep = f(input[index], index);
	state[index] = keep ? 1 : 0;
}

namespace GpuStreamCompactor
{
	uint32_t prepare(uint32_t max_element_count, uint32_t element_size, bool inplace, bool order_preserving)
	{
		//TODO: Implement me.

		uint32_t needed_size = 0;

		// here is an example of how to get memory requirements from cub
		size_t size_required;
		uint32_t* p_dev = nullptr;
		cub::DeviceReduce::Sum(nullptr, size_required, p_dev, p_dev, max_element_count);
		needed_size += size_required; // temporary cub memory
		needed_size += sizeof(uint32_t); // output of the sum
		// example of how to add another uint32 array of elements
		needed_size += max_element_count * sizeof(uint32_t);

		return needed_size;
	}

	template<typename data_type, typename compare_function>
	uint32_t run(const data_type* input, data_type* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool inplace, bool order_preserving, compare_function&& func)
	{
		//TODO: Implement me.

		// just an example

		// split the temporary memory into two arrays again via pointer arithmethics
		uint32_t* state = reinterpret_cast<uint32_t*>(temp_memory);
		uint32_t* sum_out = reinterpret_cast<uint32_t*>(temp_memory) + element_count;
		uint32_t* temp = sum_out + 1;
		uint32_t remspace = temp_size - (element_count + 1) * sizeof(uint32_t);

		// simple kernel setup
		dim3 block_size(256, 1, 1);
		dim3 blocks(divup<unsigned>(element_count, block_size.x), 1, 1);

		// note that we pass func per value here
		writeState <<<blocks, block_size>>> (input, state, element_count, func);

		// call into cub
		size_t s = remspace;
		cub::DeviceReduce::Sum(temp, s, state, sum_out, element_count);

		// copy the final count back
		uint32_t count;
		HANDLE_ERROR(cudaMemcpy((void*)&count, sum_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		return count;
	}
}
