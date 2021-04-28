#include "core/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// bert softmax code modified from Nvidia's DeepLearningExamples
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.1/fastertransformer/cuda/open_attention.cu#L1399-L1583

template <typename T>
__global__
void softmax_kernel_bert(T *qk_buf, const int64_t * pre_padding_len,const int head_num, const int seq_len, const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;
    __shared__ float s_sum, s_max;

    int left_padding_len = 0;
    if (pre_padding_len != nullptr){
        left_padding_len = pre_padding_len[batch_id];
    }

    // This is added here because if it is not added, the original value will be used by default, and the value will get bigger and bigger when there are many layers until it overflows during the calculation
    for(int i = 0; i < left_padding_len; ++i)
    {   
      if(threadIdx.x < seq_len)
          qk_buf[threadIdx.x + qk_offset] = 0.0f;
      qk_offset += seq_len;
    }

    for(int i = left_padding_len; i < seq_len; ++i)
    {   
        float qk = threadIdx.x < seq_len ? (float)qk_buf[threadIdx.x + qk_offset] : 0.0f;
        float left_padding_val = (threadIdx.x < left_padding_len)? -10000.0f:0.0f;

        float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + left_padding_val): -1e20f;

        float max_val = blockReduceMax<float>(tmp);

        if(threadIdx.x == 0)
            s_max = max_val;
        __syncthreads();

        qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

        float sum_val = blockReduceSum<float>(qk);

        if(threadIdx.x == 0)
        {
            s_sum = sum_val + 1e-6f;
        }
        __syncthreads();

        if(threadIdx.x < seq_len)
            qk_buf[threadIdx.x + qk_offset] = (T)(qk / s_sum);

        qk_offset += seq_len;
    }
}
template <class T>
void bert_softmax_kernel(void *qk_buf, const int64_t * pre_padding_len, const int &batch_size,
                         const int &head_num, const int &seq_len, const float &scalar, const cudaStream_t stream)
{
  dim3 grid, block;

  if (seq_len <= 32)
    block.x = 32;
  else if (seq_len > 32 && seq_len <= 64)
    block.x = 64;
  else if (seq_len > 64 && seq_len <= 128)
    block.x = 128;
  else if (seq_len > 128 && seq_len <= 256)
    block.x = 256;
  else if (seq_len > 256 && seq_len <= 512)
    block.x = 512;
  else
    block.x = 1024;

  grid.x = batch_size * head_num;
  softmax_kernel_bert<T><<<grid, block, 0, stream>>>((T *)qk_buf, pre_padding_len, head_num, seq_len, scalar);
}

template void bert_softmax_kernel<float>(void *qk_buf, const int64_t * pre_padding_len,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
template void bert_softmax_kernel<half>(void *qk_buf, const int64_t * pre_padding_len,const int& batch_size, 
                                      const int& head_num, const int& seq_len, const float& scalar, const cudaStream_t stream);
