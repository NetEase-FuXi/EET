#include <stdio.h>
#include <cuda_runtime.h>

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T, int item_per_thread>
__inline__ __device__
T warpReduceSum_opt(T* val)
{
    T sum_in_thread = 0;
    for (int i = 0 ; i < item_per_thread; i++){
        sum_in_thread += val[i];
    }
    for(int mask = 16; mask > 0; mask >>= 1)
        sum_in_thread += __shfl_xor_sync(FINAL_MASK, sum_in_thread, mask, 32);
    return sum_in_thread;

}
/* Calculate the sum of all elements in a block */
template <typename T, int item_per_thread>
__inline__ __device__
T blockReduceSum_opt(T* val)
{
    static __shared__ T shared[32];
    // __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    T sum_in_warp = 0;
    sum_in_warp = warpReduceSum_opt<T, item_per_thread>(val);

    if(lane == 0)
        shared[wid] = sum_in_warp;

    __syncthreads();


    //l2_val : level two of block reduce sum
    float l2_val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);

    l2_val = warpReduceSum<T>(l2_val);
    return l2_val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

__inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}



// __inline__ void sequence_kernel(int64_t* data_ptr, int64_t size) {
//   thrust::device_ptr<int64_t> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
//   thrust::sequence(thrust::device, data_dev_ptr, data_dev_ptr + size);
// }


// __inline__ void fill_kernel(int64_t* data_ptr, int64_t size, int64_t val) {
//   thrust::device_ptr<int64_t> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
//   thrust::fill(thrust::device, data_dev_ptr, data_dev_ptr + size, val);
// }

