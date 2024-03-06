// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM
#pragma once

#include <cstdint>
#include <vector>

namespace CpuStreamCompactor
{

	uint32_t prepare(uint32_t element_count, uint32_t element_size, bool inplace, bool order_preserving)
	{
		return 0;
	}

	template<typename data_type, typename compare_function>
	uint32_t run(const data_type* input, data_type* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool in_place, bool order_preserving, compare_function&& f)
	{
		uint32_t output_index = 0;
		for (uint32_t i = 0; i < element_count; i++)
		{
			if (f(input[i], i))
				output[output_index++] = input[i];
		}

		return output_index;
	}
};
// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM