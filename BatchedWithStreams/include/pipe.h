#pragma once

#include <cublas_v2.h>
#include "filter_params.h"

class Pipe 
{
    public:
        Pipe() = default;
        Pipe(
            int n,
            int m,
            int batch_size,
            FilterParams filter,
            const float* x0,
            const float* P0
        );
        Pipe(Pipe&& other);

        ~Pipe();

        void uploadFirstBatch(float* z);
        void uploadBatch(float* z);
        void xhat();
        void Phat();
        void residual();
        void gain();
        void updateX();
        void updateP();
        void downloadBatch(float* x);

    private:
        int n_;
        int m_;
        int batch_size_;
        FilterParams filter_;
        cublasHandle_t handle_;

        float* mat_1;
        float* mat_2;
        float* mat_3;
        float* mat_4;

        float** mat_3_pointers;
        float** mat_4_pointers;

        float* vec_1;
        float* vec_2;
        float* vec_3;

        int* info;

        cublasHandle_t h1;
        cublasHandle_t h2;

        cudaStream_t s1;
        cudaStream_t s2;
        cudaStream_t upload;
        cudaEvent_t e1;
        cudaEvent_t e2;
};
