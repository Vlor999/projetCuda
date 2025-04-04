#include "GPUStreamCompactor.h"
#include <cuda_runtime_api.h>
#include <iostream>

#include "DataTypes.cuh"
#include "Utility.cuh"

#include <cub/cub.cuh>

#define BLOCK_SIZE 256

template<typename T, typename F>
__global__ void writeState(const T* input, uint32_t* state, uint32_t num_elements, F f) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        state[index] = f(input[index], index);
    }
}

template<typename T>
__global__ void copyElements(const T* input, T* output, const uint32_t* state, const uint32_t* scan, uint32_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements && state[index] == 1) {
        output[scan[index]] = input[index];
    }
}

// Kernel for in-place ordered compaction
template<typename T>
__global__ void copyElementsInPlaceOrdered(T* data, const uint32_t* state, const uint32_t* scan, uint32_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements && state[index] == 1) {
        uint32_t dest_pos = scan[index];
        if (dest_pos != index) {
            data[dest_pos] = data[index];
        }
    }
}

namespace GpuStreamCompactor
{
	/*
	* Show all informations about the GPU usage
	*/
	template<typename data_type>
	void showInfo(uint32_t element_count, const data_type* input, const data_type* output, const uint32_t* state, const uint32_t* scan, cudaEvent_t start, cudaEvent_t stop){
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float time = 0;
		cudaEventElapsedTime(&time, start, stop);
		std::cout << "GPU Stream Compaction execution time: " << time << " ms" << std::endl;

		// Clean events
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		data_type* host_input = new data_type[element_count];
		uint32_t* host_state = new uint32_t[element_count];
		uint32_t* host_scan = new uint32_t[element_count];
		
		HANDLE_ERROR(cudaMemcpy(host_input, input, element_count * sizeof(data_type), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(host_state, state, element_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(host_scan, scan, element_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// Display input elements and their status
		std::cout << "Input array with keep/discard status:" << std::endl;
		int kept_count = 0;
		for (int i = 0; i < element_count; i++)
		{
			bool keep = (host_state[i] == 1);
			std::cout << "[" << i << "] " << host_input[i].toString() 
			<< " => " << (keep ? "KEPT (pos: " + std::to_string(host_scan[i]) + ")" : "DISCARDED") << std::endl;
			if (keep) kept_count++;
		}
		std::cout << "Summary: " << kept_count << " elements kept out of " << element_count << " total elements." << std::endl;

		// Free host memory
		delete[] host_input;
		delete[] host_state;
		delete[] host_scan;
	}

    uint32_t prepare(uint32_t max_element_count, uint32_t element_size, bool inplace, bool order_preserving) {
        uint32_t needed_size = 2 * max_element_count * sizeof(uint32_t) + sizeof(uint32_t);
        
        size_t scan_storage_size;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_storage_size, (uint32_t*)nullptr, (uint32_t*)nullptr, max_element_count);
        needed_size += scan_storage_size;
        
        size_t sum_storage_size;
        cub::DeviceReduce::Sum(nullptr, sum_storage_size, (uint32_t*)nullptr, (uint32_t*)nullptr, max_element_count);
        needed_size += sum_storage_size;
        
        return needed_size;
    }

    template<typename data_type, typename compare_function>
    uint32_t run(const data_type* input, data_type* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool inplace, bool order_preserving, compare_function&& func){
		// Memory :
		// [state array] [scan array] [sum result] [cub temp storage]
		// Size memory bloc : 
		// [element_count * sizeof(uint32_t)] [element_count * sizeof(uint32_t)] [sizeof(uint32_t)] [temp_size - (element_count * sizeof(uint32_t) * 2 + sizeof(uint32_t))]
		// [state array] = [scan array] = element_count * sizeof(uint32_t)
		// [sum result] = sizeof(uint32_t)
		// [cub temp storage] = temp_size - (element_count * sizeof(uint32_t) * 2 + sizeof(uint32_t))

		// Info from prepare function 

        uint8_t* mem_ptr = reinterpret_cast<uint8_t*>(temp_memory);
        uint32_t* state = reinterpret_cast<uint32_t*>(mem_ptr);
        mem_ptr += element_count * sizeof(uint32_t);
        
        uint32_t* scan = reinterpret_cast<uint32_t*>(mem_ptr);
        mem_ptr += element_count * sizeof(uint32_t);
        
        uint32_t* sum_out = reinterpret_cast<uint32_t*>(mem_ptr);
        mem_ptr += sizeof(uint32_t);
        
        void* cub_temp_storage = mem_ptr;
        size_t cub_temp_storage_bytes = temp_size - (mem_ptr - reinterpret_cast<uint8_t*>(temp_memory));

        dim3 block_size(BLOCK_SIZE, 1, 1);
        dim3 blocks((element_count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        writeState<<<blocks, block_size>>>(input, state, element_count, func);
        cub::DeviceReduce::Sum(cub_temp_storage, cub_temp_storage_bytes, state, sum_out, element_count);
        cub::DeviceScan::ExclusiveSum(cub_temp_storage, cub_temp_storage_bytes, state, scan, element_count);

        uint32_t count;
        HANDLE_ERROR(cudaMemcpy(&count, sum_out, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (inplace) {
            if (order_preserving){
                copyElementsInPlaceOrdered<<<blocks, block_size>>>(output, state, scan, element_count);
            } 
			else if (count > 0){
                data_type* temp_buffer;
                HANDLE_ERROR(cudaMalloc(&temp_buffer, element_count * sizeof(data_type)));
                HANDLE_ERROR(cudaMemcpy(temp_buffer, input, element_count * sizeof(data_type), cudaMemcpyDeviceToDevice));
                copyElements<<<blocks, block_size>>>(temp_buffer, output, state, scan, element_count);
                HANDLE_ERROR(cudaFree(temp_buffer));
            }
        } 
		else {
            copyElements<<<blocks, block_size>>>(input, output, state, scan, element_count);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

		#ifdef DEBUG_MODE
        GPUStreamCompactor::showInfo(element_count, input, output, state, scan, start, stop);
        #endif

        return count;
    }
}
