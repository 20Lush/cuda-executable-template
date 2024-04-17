#include "kernel.hpp"

#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdexcept>

namespace lush::cuda {

    // CUDA Kernel
    __global__ void VecAdd_device(float* A, float* B, float* C) {
        
        // Grab the thread idx for this job
        int i = threadIdx.x;

        // Perform the kernel execution on this thread idx
        C[i] = A[i] + B[i];

    }

    // C++ interface, stl compliant. Could be inlined, but gcc is probably already doing that anyway. We handle all of our host->device work here.
    std::vector<float> VecAdd(int blocks, int threads_per_block, std::vector<float>& A, std::vector<float>& B) {

        // Check an edge case that could cause all sorts of chaos
        if(A.size() != B.size()) {
            throw std::runtime_error("Error: Cannot add vectors of different length. Returning zeros");
            return std::vector<float>(A.size(), 0.0);
        }
        
        // STL->thrust vector type, able to be easily allocated in the GPU
        thrust::device_vector<float> A_device = A;
        float* A_device_ptr = thrust::raw_pointer_cast(A_device.data());

        // STL->thrust vector type, able to be easily allocated in the GPU
        thrust::device_vector<float> B_device = B;
        float* B_device_ptr = thrust::raw_pointer_cast(B_device.data());

        // thrust vector of the size of our parameters that we will dump the output of the kernel to
        thrust::device_vector<float> sum(A_device.size(), 0.0);
        float* sum_device_ptr = thrust::raw_pointer_cast(sum.data());

        // Kernel call. Blocks from pameter, TpB from parameter, pass in pointers to the nice containerized thrust vectors
        VecAdd_device<<<blocks, threads_per_block>>>(A_device_ptr, B_device_ptr, sum_device_ptr);

        // Wait for everone to finish
        cudaDeviceSynchronize();

        // Create our STL return type with the same size as the device types. SILENTLY FAILS IF size1 != size2
        std::vector<float> stl_rtn(sum.size());

        // Copy from the device container to the STL host type.
        thrust::copy(sum.begin(), sum.end(), stl_rtn.begin());

        return stl_rtn;
    }

}
