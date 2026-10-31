#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Бин поиск по диагонали d, т.е. кол-во единиц на диагонали. При условии, что под Merge Path единицы – над нули.
__device__ int merge_path(
    const unsigned int* A, unsigned int na,
    const unsigned int* B, unsigned int nb,
    unsigned int d)
{
    unsigned int low  = d > nb ? d - nb : 0;
    unsigned int high = d > na ? na : d;

    while (low < high) {
        unsigned int a = (low + high) / 2;
        unsigned int b = d - a;
        if (A[a] <= B[b - 1]) {
            low = a + 1;
        } else {
            high = a;
        }
    }

    return low;
}

__global__ void block_sort(
    const unsigned int* input_data,
    unsigned int* output_data, int n)
{
    __shared__ unsigned int shared0[BLOCK_SORT_SIZE];
    __shared__ unsigned int shared1[BLOCK_SORT_SIZE];

    const int tile_base = blockIdx.x * BLOCK_SORT_SIZE;
    const int cnt = min(BLOCK_SORT_SIZE, n - tile_base);
    const int tid = threadIdx.x;
    const int V = MERGE_VALUES_PER_THREAD;

    unsigned int* src = shared0;
    unsigned int* dst = shared1;

    #pragma unroll
    for (int j = tid; j < cnt; j += blockDim.x)
        src[j] = input_data[tile_base + j];

    __syncthreads();

    for (int k = 1; k < cnt; k <<= 1) {
        const int k2 = 2 * k;
        const int q = (k2 + V - 1) / V;
        const int pairs = (cnt + k2 - 1) / k2;

        for (int task = tid; task < pairs * q; task += blockDim.x) {
            const int base = (task / q) * k2;
            const int mid = min(base + k, cnt);
            const int end = min(base + k2, cnt);

            const int d0 = (task % q) * V;
            if (d0 >= end - base)
                continue;

            const int d1 = min(d0 + V, end - base);

            const unsigned int* A = src + base;
            const unsigned int* B = src + mid;
            const int na = mid - base;
            const int nb = end - mid;

            int a = merge_path(A, na, B, nb, d0);
            int b = d0 - a;

            for (int d = d0; d < d1; ++d) {
                if (b == nb || (a < na && A[a] <= B[b])) {
                    dst[base + d] = A[a++];
                } else {
                    dst[base + d] = B[b++];
                }
            }
        }

        __syncthreads();
        unsigned int* tmp = src;
        src = dst;
        dst = tmp;
    }

    for (int j = tid; j < cnt; j += blockDim.x)
        output_data[tile_base + j] = src[j];
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int V = MERGE_VALUES_PER_THREAD;
    const int k2 = 2 * sorted_k;
    const int q = (V - 1 + k2) / V; // ceil(2k / V)
    const int pairs = (n + k2 - 1) / k2; // ceil(n / 2k)

    if (i >= pairs * q)
        return;

    const int pair = i / q;
    const int chunk = i % q;

    const int base = pair * k2;
    const int mid = min(base + sorted_k, n);
    const int end = min(base + k2, n);

    const int d0 = chunk * V;
    if (d0 >= end - base)
        return;

    const int d1 = min(d0 + V, end - base);

    const unsigned int* A = input_data + base;
    const int na = mid - base;
    const unsigned int* B = input_data + mid;
    const int nb = end - mid;

    int a = merge_path(A, na, B, nb, d0);
    int b = d0 - a;

    for (int d = d0; d < d1; ++d) {
        if (b == nb || (a < na && A[a] <= B[b])) {
            output_data[base + d] = A[a++];
        } else {
            output_data[base + d] = B[b++];
        }
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}

void block_sort(const gpu::gpu_mem_32u& input_data,
                gpu::gpu_mem_32u& output_data, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const int blocks = (n + BLOCK_SORT_SIZE - 1) / BLOCK_SORT_SIZE;
    ::block_sort<<<blocks, GROUP_SIZE, 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
