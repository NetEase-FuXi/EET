#ifndef _MMANAGER_
#define _MMANAGER_
#include "op/common.hpp"
#include "vector"
#include <cuda_fp16.h>

#define MAX_BUFFER_SIZE 100

namespace eet{
    //buffer for workers
    class Buffer{
        public:
        Buffer(const int& size, const c10::ScalarType& dtype, const torch::TensorOptions options):
                size_(size), dtype_(dtype), options_(options){
            tensor_ = torch::zeros(size, options_);
            is_idle_ = false;
        }

        void* data_ptr() const{
            if (tensor_.data_ptr() == nullptr){
                printf("Error : try to use a empty Buffer \n");
            }
            return this->tensor_.data_ptr();
        }

        torch::Tensor& get_tensor(){
            return tensor_;
        } 

        //now we ignore the compare for dtype
        bool is_ok(const int& size, const c10::ScalarType& dtype) const {
                        if (!is_idle_){
                            return false;
                        }
                        if (size == size_){
                            return true;
                        }else{
                            return false;
                        }
                    }
        void set_busy(){
            is_idle_ = false;
        }
        void free(){
            is_idle_ = true;
        }
        bool is_idle(){
            return is_idle_;
        }
        void print(const bool verbose = false, const int& line = 4) const {
            if (dtype_ == torch::kFloat32){
                printf(" [ EET Buffer, Fp32, %ld ]\n ", size_);
                if (verbose){
                    float* tmp = (float*)malloc(sizeof(float) * size_);
                    check_cuda_error(cudaMemcpy(tmp, tensor_.data_ptr(), sizeof(float) * size_, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < size_; i++){
                        printf("%f ", tmp[i]);
                        if (i % line == 0){
                            printf("\n");
                        }
                    }
                }
            } else{
                printf(" [ EET Buffer, Fp16, %ld ] \n", size_);
                if (verbose){
                    half* tmp = (half*)malloc(sizeof(half) * size_);
                    check_cuda_error(cudaMemcpy(tmp, tensor_.data_ptr(), sizeof(half) * size_, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < size_; i++){
                        printf("%f ", __half2float(tmp[i]));
                        if (i % line == 0){
                            printf("\n");
                        }
                    }
                }
            }   
        }
        ~Buffer(){
        }
        Buffer() = delete;
        Buffer(const Buffer&){
            printf("[EET][ERROR] Buffer capacity is over, increse macro MAX_BUFFER_SIZE to solve it \n");
            exit(0);
        }
        Buffer& operator = (const Buffer&) = delete;

        private:
            bool is_idle_;
            long size_;
            c10::ScalarType dtype_;
            torch::TensorOptions options_;
            torch::Tensor tensor_;
    };

    // memory manager
    class MManager{
        public:
        MManager(){
           buffers_.reserve(MAX_BUFFER_SIZE);
        }
        Buffer& get_buffer(const int& size, const c10::ScalarType& dtype, const torch::TensorOptions& options){
                    for (auto& buffer : buffers_){
                        if (buffer.is_ok(size, dtype)){
                            buffer.set_busy();
                            return buffer;
                        }
                    }
                    std::cout << "There are " << buffers_.size() << " buffer in vector" << std::endl;

                    buffers_.emplace_back(size, dtype, options);
                    return buffers_.back();
        }
        static MManager& get_instance(){
            return *manager;
        }
        ~MManager(){
            buffers_.clear();
        }
        private:
            static MManager* manager;
            std::vector<Buffer> buffers_;
    };
}
#endif