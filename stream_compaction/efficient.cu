#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

# define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Upsweep on kernel.
        __global__ void kernUpsweep(int n, int d, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            k *= (1 << (d + 1));

            if (k >= n) {
                return;
            }

            idata[k + (1 << (d + 1)) - 1] += idata[k + (1 << d) - 1];
        }

        // Downsweep on kernel.
        __global__ void kernDownsweep(int n, int d, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            k *= (1 << (d + 1));

            if (k >= n) {
                return;
            }

            int temp = idata[k + (1 << d) - 1]; // Node.leftChild save.
            idata[k + (1 << d) - 1] = idata[k + (1 << (d + 1)) - 1]; // Node.leftChild = Node.val.
            idata[k + (1 << (d + 1)) - 1] += temp; // Node.rightChild = Node.leftChild + Node.val.
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // One buffer for in place scan.
            int* dev_buffer;

            // Set new n w/ log2ceil function.
            int round_n = 1 << ilog2ceil(n);

            // Allocate buffer.
            cudaMalloc((void**)&dev_buffer, round_n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMemset(dev_buffer, 0, round_n * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO

            // Upsweep.
            for (int d = 0; d < ilog2ceil(n); d++) {
                // Call upsweep kern function w/ rounded n val.
                kernUpsweep << < fullBlocksPerGrid, blockSize >> > (round_n, d, dev_buffer);
                checkCUDAError("kernUpsweep failed.");
            }

            // Downsweep. 
            // Set (round_n - 1) val in dev_buffer = 0.
            cudaMemset(dev_buffer + (round_n - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n); d >= 0; d--) {
                // Call downsweep kern func.
                kernDownsweep << < fullBlocksPerGrid, blockSize >> > (round_n, d, dev_buffer);
            }
;           timer().endGpuTimer();

            // Copy data back to host from dev_buffer.
            cudaMemcpy(odata, dev_buffer, round_n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_buffer);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
