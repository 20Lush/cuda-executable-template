#include "main.hpp"
#include "kernel.hpp"

#include <cuda_runtime.h>


// Make sure to change the namespace EXECUTABLE to something more specific
namespace lush::EXECUTABLE {

	// Any forward declerations?

	bool print_args(int argc, char** argv) {

		std::cout << "args: ";
		for (int idx = 0; idx < argc; idx++) {
			std::cout << argv[idx] << " ";
		}
		std::cout << std::endl;

		return true;
	}

}  // namespace lush::EXECUTABLE

int main(int argc, char** argv) {

	std::cout << "Hello World! Executable given " << argc << " arguments." << std::endl;
	lush::EXECUTABLE::print_args(argc, argv);

	std::vector<float> A = {1.0, 1.0, 1.0};
	std::vector<float> B = {2.0, 2.0, 2.0};

	const int blocks = 1;

	std::vector<float> sum = lush::cuda::VecAdd(blocks, A.size(), A, B);

	std::cout << "GPU VecAdd has result: " << sum[0] << ", " << sum[1] << ", " << sum[2] << std::endl;

}