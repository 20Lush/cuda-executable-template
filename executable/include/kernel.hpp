#pragma once
#include <vector>

#include <thrust/host_vector.h>

namespace lush::cuda {

	/** @brief Add two vectors of any size on the GPU
	 *  @returns Outputs sum into vector Y
	 *	@throws Runtime exception when A.size() and B.size() are not equal.
	 */ 
	std::vector<float> VecAdd(int blocks, int threads_per_block, std::vector<float>& A, std::vector<float>& B);

}  // namespace lush::cuda