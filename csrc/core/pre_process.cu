#include "core/common.cuh"
#include <cuda_fp16.h>
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

__global__
void gen_mask_offset(const int64_t *every_seq_len, const int current_batch_num, int64_t *dest) {
    const int tid = threadIdx.x;
    extern __shared__ __align__(sizeof(int)) int s_every_len[];

    s_every_len[tid] = every_seq_len[tid];
    int sum_of_previous_seq = 0;
    int current_len = s_every_len[tid];
    for (int i = 0; i < tid; i++) {
        sum_of_previous_seq += s_every_len[i];
    }
    for (int i = 0; i < current_len; i++) {
        dest[sum_of_previous_seq + i] = sum_of_previous_seq;
    }
}


void fill_kernel(int64_t* data_ptr, int64_t size, int64_t val) {
  thrust::device_ptr<int64_t> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::fill(thrust::device, data_dev_ptr, data_dev_ptr + size, val);
}


int reduce_kernel(int64_t* data_ptr, int64_t size) {
  thrust::device_ptr<int64_t> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  return thrust::reduce(thrust::device, data_dev_ptr, data_dev_ptr + size, 0);
}

void gen_mask_offset_kernel(const int64_t* every_seq_len_inbatch, const int current_batch_num, int64_t* dest, const cudaStream_t stream){
    dim3 grid(1);
    assert(current_batch_num <= 1024);
    dim3 block(current_batch_num);
    int shared_mem_size = current_batch_num * sizeof(int);
    gen_mask_offset<<<grid, block, shared_mem_size, stream>>>(every_seq_len_inbatch, current_batch_num, dest);
}

__global__
void  compute_len_inbatch(int64_t* data_ptr, int batch_size, int seq_len, int64_t* dest){

    int64_t *data = data_ptr + blockIdx.x * seq_len;
    thrust::device_ptr<int64_t> data_dev_ptr = thrust::device_pointer_cast(data);
    dest[blockIdx.x] = thrust::reduce(thrust::device, data, data + seq_len, 0);
}

void compute_len_inbatch_kernel(int64_t *data_ptr, int batch_size,int seq_len, int64_t *dest, const cudaStream_t stream){
  dim3 grid(batch_size);
  dim3 block(1);
  compute_len_inbatch<<<grid, block,0, stream>>>(data_ptr, batch_size,seq_len,dest);
}

void gen_mask_offset_kernel(const int64_t *every_seq_len_inbatch,
                                    const int current_batch_num, int64_t *dest, const cudaStream_t stream);

void fill_kernel(int64_t *data_ptr, int64_t size, int64_t val);

int reduce_kernel(int64_t *data_ptr, int64_t size);

void compute_len_inbatch_kernel(int64_t *data_ptr, int batch_size, int seq_len, int64_t *dest, const cudaStream_t stream);
