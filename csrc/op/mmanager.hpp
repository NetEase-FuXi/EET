#ifndef _MMANAGER_
#define _MMANAGER_
#include "op/common.hpp"
#include "vector"
#include <cuda_fp16.h>
#include <unordered_map>

#define MAX_BUFFER_SIZE 100
#define MAX_CACHE_SIZE 8 //cache_is dedicated to storing output, there are several modules, so the number is relatively small
namespace eet{
    //buffer for workers
    class Buffer{
        public:
        Buffer(const int& size, const c10::ScalarType& dtype, const torch::TensorOptions options,std::string str = "no_name_"):
                size_(size), dtype_(dtype), options_(options){

            tensor_ = torch::ones(size, options_).contiguous();
            is_idle_ = false;
            strs_.push_back(str);
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
        bool is_ok(const int &size, const c10::ScalarType &dtype) const
        {
            if (!is_idle_)
            {
                return false;
            }
            if (size == size_)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        bool check_str(const std::string& str) const
        {
            for (auto& str_ : strs_) {
                if (str.compare(str_) != 0) {
                    return false;
                } else {
                    return true;
                }
            }
        }

        void register_str(const std::string& str){
            if (std::find(strs_.begin(), strs_.end(),str) == strs_.end()) {
                strs_.push_back(str);
            }
        }

        std::vector<std::string> get_strs() const{
            return strs_;
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
                    delete tmp;
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
                    delete tmp;
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
            std::vector<std::string> strs_;
    };

    // memory manager
    class MManager{
        public:
        MManager(){
           buffers_.reserve(MAX_BUFFER_SIZE);
           cache_.reserve(MAX_CACHE_SIZE);
        }

        void report_buffer(){
            for (auto& buf : buffers_){
                for (auto& str : buf.get_strs()) {
                    std::cout << str << " -->  ";
                }
                std::cout << buf.get_tensor().sizes() << std::endl;
            }
        }

        void report_cache(){
            for (auto& buf : cache_){
                for (auto& str : buf.get_strs()) {
                    std::cout << str << " -->  ";
                }
                std::cout << buf.get_tensor().sizes() << std::endl;
            }
        }

        //strict : true --> size must equal, then reuse
        //        false --> size smaller, then reuse
        Buffer& get_buffer(const size_t& size, const c10::ScalarType& dtype,
                           const torch::TensorOptions& options,
                           const std::string& name = "no_name",
                           bool strict = true){
                    int itemsize = 0;
                     switch (dtype) {
                         case torch::kFloat32:
                             itemsize = 4;
                             break;
                         case torch::kFloat16:
                             itemsize = 2;
                             break;
                             //TODO
                         case torch::kInt8:
                             itemsize = 1;
                             break;
                     }
                    for (auto& buffer : buffers_){
                        if ((buffer.is_ok(size, dtype) && strict) ||
                                (buffer.get_tensor().nbytes() >= size * itemsize && !strict)){
                            buffer.set_busy();
                            buffer.register_str(name);
                            return buffer;
                        }
                    }
                    std::cout << "There are " << buffers_.size() << " buffer in vector" << std::endl;
                    std::cout << "Request a buffer of size : " << size << std::endl;

                    buffers_.emplace_back(size, dtype, options,name);
                    return buffers_.back();
        }

        Buffer &get_cache(const size_t &size, const c10::ScalarType &dtype, const torch::TensorOptions &options, std::string str)
        {
            for (auto& cache : cache_){
                if (cache.check_str(str)){
                    cache.register_str(str);
                    return cache;
                }
            }
            std::cout << "There are " << cache_.size() << " buffer in cache vector" << std::endl;
            std::cout << "Request a cache of size : " << size << std::endl;

            cache_.emplace_back(size, dtype, options,str);
            return cache_.back();
        }

        static MManager& get_instance(){
            return *manager;
        }

        void clear(){
            buffers_.clear();
            cache_.clear();
        }

        ~MManager(){
            buffers_.clear();
            cache_.clear();
        }
        private:
            static MManager* manager;
            std::vector<Buffer> buffers_;
            std::vector<Buffer> cache_;
    };
}
#endif