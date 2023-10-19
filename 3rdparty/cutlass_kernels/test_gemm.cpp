#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <vector>
#include "fpA_intB_gemm.h"


void weight_only_int8_mul(half *input, int8_t * weight, half* scale, half* output, int m, int n, int k)
{
    fastertransformer::gemm_fp16_int_bias_act(
        reinterpret_cast<fastertransformer::half *>(input),
        reinterpret_cast<const uint8_t*>(weight),
        reinterpret_cast<fastertransformer::half *>(scale),
        nullptr,
        reinterpret_cast<fastertransformer::half *>(output),
        std::nullopt,
        m, k, n,
        0,
        nullptr,
        0,
        0
        );
}


int main(){
    using T = half;
    int M = 1;
    int N = 1;

    T* input_host = (T*)malloc(M*N*sizeof(T));
    T* output_host = (T*)malloc(M*N*sizeof(T));
    int8_t* weight = (int8_t*)malloc(N*N*sizeof(int8_t));
    T* weight_scale = (T*)malloc(N*sizeof(T));
    for (int i = 0; i < N; i++) {
        input_host[i] = T(i % 16 + 1.0);
        weight[i] = i % 16;
        weight_scale[i] = T(1.0);
    }
    T *input_device;
    T *output_device;
    int8_t *weight_device;
    T *weight_scale_device;
    cudaMalloc((void **)&input_device, N*sizeof(T));
    cudaMalloc((void **)&output_device, N*sizeof(T));
    cudaMalloc((void **)&weight_device, N*sizeof(int8_t));
    cudaMalloc((void **)&weight_scale_device, N*sizeof(T));

    cudaMemcpy(input_device, input_host, N*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, N*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_scale_device, weight_scale, N*sizeof(T), cudaMemcpyHostToDevice);

    weight_only_int8_mul(input_device, weight_device, weight_scale_device, output_device, 1, 1, 1);

    cudaMemcpy(output_host, output_device, N * sizeof(T), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     printf("%.5f\n", (float)output_host[i]);
    // }

    int iter = 11;
    std::vector<float> results;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    for (int i = 0; i < iter; i++) {
        float duration_ms;
        cudaEventRecord(start_event, 0);

        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&duration_ms, start_event, stop_event);
        results.emplace_back(duration_ms);
    }
    std::sort(results.begin(), results.end());

    printf("median %.2f ms\n", results[iter / 2]);

    cudaFree(input_device);
    cudaFree(output_device);
    cudaFree(weight_device);
    cudaFree(weight_scale_device);
    free(input_host);
    free(output_host);
    free(weight);
    free(weight_scale);
    return 0;
}