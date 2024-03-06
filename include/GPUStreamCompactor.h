// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM
#pragma once

#include <cstdint>

namespace GpuStreamCompactor
{
	/// <summary>
	/// This method is used to determine the needed additional gpu device memory.
	/// It is called during preparation.
	/// </summary>
	/// <param name="max_element_count">The maximum number of elements in the input array</param>
	/// <param name="element_size">The element size in bytes</param>
	/// <param name="inplace">If true, the input and output pointers will be the same</param>
	/// <param name="order_preserving">If true, after compaction, the elements must retain the same order as they appeared in the input stream</param>
	uint32_t prepare(uint32_t max_element_count, uint32_t element_size, bool inplace, bool order_preserving);

	/// <summary>
	/// Executes the stream compaction.
	/// </summary>
	/// <param name="input">The device input array on which the compaction operation is executed</param>
	/// <param name="output">The device output array holds the result of the compaction operation (unless it's an in-place operation)</param>
	/// <param name="element_count">The number of elements in the input array</param>
	/// <param name="temp_memory">Additional device memory allocated based on the size obtained from the prepare method</param>
	/// <param name="temp_size">Size of the additional memory</param>
	/// <param name="inplace">After compaction, the input array holds the result of the compaction operation</param>
	/// <param name="order_preserving">After compaction, the elements retain the same order as they appeared in the input stream</param>
	/// <param name="compare">Callable to be executed for each input element, returning true if it should be retained; signature: bool functor(const data_type& element, uint32_t index)</param>
	template<typename data_type, typename compare_function>
	uint32_t run(const data_type* input, data_type* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool inplace, bool order_preserving, compare_function&& func);
}
// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM