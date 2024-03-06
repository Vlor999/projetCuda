// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM

#include <cstdint>
#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <unordered_map>
#include <random>

#include <cuda_runtime_api.h>

#include "CPUTimer.h"
#include "GPUTimer.cuh"

#include "Random.hpp"
#include "DataTypes.cuh"

#include "CPUStreamCompactor.h"
#include "GPUStreamCompactor.h"

// only two tests in the fraemwork
uint32_t RunGPUTest(const TestType* input, TestType* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool inplace, bool order_preserving, int test);
uint32_t RunCPUTest(const TestType* input, TestType* output, uint32_t element_count, void* temp_memory, uint32_t temp_size, bool inplace, bool order_preserving, int test)
{
	if (test == 0)
		return CpuStreamCompactor::run(input, output, element_count, temp_memory, temp_size, inplace, order_preserving, test_hash0);
	if (test == 1)
		return CpuStreamCompactor::run(input, output, element_count, temp_memory, temp_size, inplace, order_preserving, test_hash1);
	if (test == 2)
		return CpuStreamCompactor::run(input, output, element_count, temp_memory, temp_size, inplace, order_preserving, test_hash2);
	return 0xFFFFFFFF;
}

template<typename T>
std::unordered_map<T, uint32_t> createElementList(const std::vector<T>& vec, uint32_t element_count)
{
	std::unordered_map<T, uint32_t> element_map;
	// Count elements for non preserving order check.
	for (size_t i = 0; i < element_count; i++)
	{
		element_map[vec[i]]++;
	}
	return element_map;
}


int main(int argc, char* argv[])
{
	std::cout << "Assignment 01 - Stream Compaction" << std::endl;

	size_t runs = 3;
	bool runCPU = true;
	int num_elements = 10000000;
	int seed = 0x1234;
	int test = 0;
	bool order_preserving = true;
	bool inplace = false;
	try
	{
		if (argc > 1)
			runs = std::atoi(argv[1]);
		if (argc > 2)
			num_elements = std::atoi(argv[2]);
		if (argc > 3)
			seed = std::atoi(argv[3]);
		if (argc > 4)
			runCPU = std::atoi(argv[4]) != 0;
		if (argc > 5)
			order_preserving = std::atoi(argv[5]) != 0;
		if (argc > 6)
			inplace = std::atoi(argv[6]) != 0;
		if (argc > 7)
			test = std::atoi(argv[7]);

		std::string seed_string = std::to_string(seed);
		std::cout << "Running Seed " << seed_string
			<< ", test=" << test
			<< ", elements=" << num_elements
			<< ", order preserving=" << order_preserving
			<< ", inplace=" << inplace << std::endl;

		using DataType = TestType;

		Random::Seed(seed);

		std::vector<DataType> cpu_input[2] = { std::vector<DataType>(num_elements), std::vector<DataType>(num_elements) };
		for (auto& v : cpu_input)
			for (auto& e : v)
				e.initRandom();

		std::vector<DataType> cpu_output[2] = { std::vector<DataType>(num_elements), std::vector<DataType>(num_elements) };
		uint32_t cpu_element_count[2] = { 0u, 0u };
		uint32_t gpu_element_count[2] = { 0u, 0u };

		// CPU.
		if (runCPU)
		{
			CPUTimer cputimer(static_cast<int>(runs));
			std::cout << "Profiling " << runs << " run(s) on the CPU" << std::endl;

			for (size_t i = 0; i < runs; i++)
			{
				auto el = std::min<size_t>(i, 1);
				cputimer.start();
				// we actually never run this inplace.. but we know the algorithm behind..
				cpu_element_count[el] = RunCPUTest(cpu_input[el].data(), cpu_output[el].data(), static_cast<uint32_t> (cpu_input[el].size()), nullptr, 0, inplace, order_preserving, test);
				cputimer.end();
			}

			std::cout << "Number of elements after compaction (CPU): " << cpu_element_count[0] << "; " << cpu_element_count[1] << std::endl;

			auto cpures = cputimer.generateResult();
			std::cout << "CPU required " << cpures.mean_ << "ms on average with std " << cpures.std_dev_ << "ms on the CPU" << std::endl;
		}

		std::unordered_map<DataType, uint32_t> cpu_element_map[2] = { createElementList(cpu_output[0], cpu_element_count[0]), 
			                                                          createElementList(cpu_output[1], cpu_element_count[1]) };

		// GPU.
		std::cout << "Profiling " << runs << " run(s) on the GPU" << std::endl;
		int failedRuns = 0;

		DataType* d_input_original[2];
		DataType* d_input;
		DataType* d_output;
		void* d_temporary = nullptr;
		uint32_t temp_memory_size = GpuStreamCompactor::prepare(static_cast<uint32_t> (num_elements), sizeof(DataType), inplace, order_preserving);
		HANDLE_ERROR(cudaMalloc((void**)&d_input_original[0], num_elements * sizeof(DataType)));
		HANDLE_ERROR(cudaMalloc((void**)&d_input_original[1], num_elements * sizeof(DataType)));
		HANDLE_ERROR(cudaMalloc((void**)&d_input, num_elements * sizeof(DataType)));
		HANDLE_ERROR(cudaMalloc((void**)&d_output, num_elements * sizeof(DataType)));
		if (temp_memory_size > 0u)
			HANDLE_ERROR(cudaMalloc((void**)&d_temporary, temp_memory_size));
		HANDLE_ERROR(cudaMemcpy(d_input_original[0], cpu_input[0].data(), num_elements * sizeof(DataType), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_input_original[1], cpu_input[1].data(), num_elements * sizeof(DataType), cudaMemcpyHostToDevice));

		GPUTimer gpu_timer(static_cast<int>(runs));
		CPUTimer cpu_timer(static_cast<int>(runs));

		for (size_t i = 0; i < runs; ++i)
		{
			auto el = std::min<size_t>(i, 1);
			HANDLE_ERROR(cudaMemcpy(d_input, d_input_original[el], num_elements * sizeof(DataType), cudaMemcpyDeviceToDevice));

			cpu_timer.start();
			gpu_timer.start();
			gpu_element_count[el] = RunGPUTest(d_input, inplace ? d_input : d_output, static_cast<uint32_t> (num_elements), d_temporary, temp_memory_size, inplace, order_preserving, test);
			gpu_timer.end();
			cpu_timer.end();

			// Evaluation.
			if (i < 2 && runCPU)
			{
				DataType* d_output_to_copy = inplace ? d_input : d_output;
				std::vector<DataType> gpu_output(num_elements);
				HANDLE_ERROR(cudaMemcpy(gpu_output.data(), d_output_to_copy, gpu_output.size() * sizeof(DataType), cudaMemcpyDeviceToHost));
				auto gpu_element_map = createElementList(gpu_output, gpu_element_count[el]);

				if (cpu_element_count[i] != gpu_element_count[i])
				{
					// ERROR: GPU elemnt count does not match the CPU element count!
					failedRuns++;
					continue;
				}

				if (order_preserving)
				{
					for (size_t k = 0; k < cpu_element_count[i]; k++)
					{
						if (cpu_output[i][k] != gpu_output[k])
						{
							// ERROR: GPU does not match CPU elements!
							failedRuns++;
							continue;
						}
					}
				}
				else
				{
					if (gpu_element_map != cpu_element_map[i])
					{
						// ERROR: GPU does not match CPU elements!
						failedRuns++;
						continue;
					}
				}
			}
		}

		std::cout << "Number of elements after compaction (GPU): " << gpu_element_count[0] << "; " << gpu_element_count[1] << std::endl;

		HANDLE_ERROR(cudaFree(d_input_original[0]));
		HANDLE_ERROR(cudaFree(d_input_original[1]));
		HANDLE_ERROR(cudaFree(d_input));
		HANDLE_ERROR(cudaFree(d_output));
		if (temp_memory_size > 0u)
			HANDLE_ERROR(cudaFree(d_temporary));

		if (runCPU)
		{
			if (failedRuns == 0)
			{
				std::cout << " SUCCESS ";
			}
			else
			{
				std::cout << " FAILED ";
			}
		}

		auto cpures = cpu_timer.generateResult();
		auto gpures = gpu_timer.generateResult();
		std::cout << "GPU required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the GPU (" << cpures.mean_ << " +/-" << cpures.std_dev_ << "ms on the CPU)" << std::endl;

		// Write to output file
		std::ofstream results_csv;
		results_csv.open("results.csv", std::ios_base::app);
		results_csv << seed_string << "," << gpures.mean_ << "," << temp_memory_size << "," << (failedRuns == 0 ? "1" : "0") << std::endl;
		results_csv.close();
	}
	catch (std::exception& ex)
	{
		std::cout << "Error: " << ex.what();
		return -1;
	}

	return 0;
}

// DO NOT CHANGE THIS FILE, THIS IS REPLACED ON THE TEST SYSTEM