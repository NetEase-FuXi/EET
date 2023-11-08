#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
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

template <typename T, int item_per_thread>
__inline__ __device__
T warpReduceMax_opt(T* val)
{
    T max_val = val[0];
    for (int i = 0 ; i < item_per_thread; i++){
        max_val = val[i] > max_val?val[i]:max_val;
    }
    for(int mask = 16; mask > 0; mask >>= 1)
        max_val = max(max_val, __shfl_xor_sync(FINAL_MASK, max_val, mask, 32));
    return max_val;

}
/* Calculate the sum of all elements in a block */
template <typename T, int item_per_thread>
__inline__ __device__
T blockReduceMax_opt(T* val)
{
    static __shared__ T shared[32];
    // __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    T sum_in_warp = 0;
    sum_in_warp = warpReduceMax_opt<T, item_per_thread>(val);

    if(lane == 0)
        shared[wid] = sum_in_warp;

    __syncthreads();


    float l2_val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;

    l2_val = warpReduceMax<T>(l2_val);
    return l2_val;
}

// Vec_n_t
template<typename T, int N>
struct GetVecType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using VecType = typename GetVecType<T, N>::type;

template<typename T, int N>
union Vec_n_t {
  static_assert(sizeof(VecType<T, N>) == sizeof(T) * N, "");
  __device__ Vec_n_t() {
    // do nothing
  }
  VecType<T, N> storage;
  T elem[N];
};

__inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id2 * (dim_3 * dim_4) + id3 * dim_4 + id4;
}

inline __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step)
{
    const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ half2 rotary_embedding_transform(const half2 v, const float2 coef)
{
    float2 fv     = __half22float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return 	__float22half2_rn(rot_fv);
}

inline __device__ void apply_rotary_embedding(half2& q, half2& k, int tid, int embed_dim, int t_step)
{
    if (2 * tid >= embed_dim) {
        return;
    }
    half2 temp = q;
    const auto coef = rotary_embedding_coefficient(2 * tid, embed_dim, t_step);
    q               = rotary_embedding_transform(temp, coef);
    // printf("tid: %d, q: %f, %f, q_embed: %f, %f, coef: %f, %f,\n", tid, __half2float(temp.x), __half2float(temp.y), __half2float(q.x), __half2float(q.y), coef.x, coef.y);
    k               = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(half& q1, half& q2, half& k1, half& k2, int tid, int embed_dim, int t_step)
{
    if (2 * tid >= embed_dim) {
        return;
    }
    float q1f = __half2float(q1);
    float q2f = __half2float(q2);
    float k1f = __half2float(k1);
    float k2f = __half2float(k2);
    const auto coef = rotary_embedding_coefficient(2 * tid, embed_dim, t_step);
    q1 = __float2half(q1f * coef.x - q2f * coef.y);
    q2 = __float2half(q2f * coef.x + q1f * coef.y);
    k1 = __float2half(k1f * coef.x - k2f * coef.y);
    k2 = __float2half(k2f * coef.x + k1f * coef.y);
    // if (tid == 0 || t_step == 0)
    //   printf("tid: %d, q: %f, %f, q_embed: %f, %f, coef: %f, %f,\n", tid, q1f, q2f, __half2float(q1), __half2float(q2), coef.x, coef.y);
}


inline __device__ void apply_rotary_embedding(float& q1, float& q2, float& k1, float& k2, int tid, int embed_dim, int t_step)
{
    if (2 * tid >= embed_dim) {
        return;
    }

    const auto coef = rotary_embedding_coefficient(2 * tid, embed_dim, t_step);
    q1 = q1 * coef.x - q2 * coef.y;
    q2 = q2 * coef.x + q1 * coef.y;
    k1 = k1 * coef.x - k2 * coef.y;
    k2 = k2 * coef.x + k1 * coef.y;
}

