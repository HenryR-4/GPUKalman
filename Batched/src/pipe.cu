#include "pipe.h"

#include <cublas_v2.h>
#include <helper_cuda.h>

namespace
{
    const float one = 1.0f;
    const float minus_one = -1.0f;
    const float zero = 0.0f;

    int addToBatchBlockSize = 32;
    __global__
    void addToBatch(float* batch, const float* to_add, int size, int num_batches)
    {
        for (
            int i = threadIdx.x + blockDim.x*blockIdx.x;
            i < size*num_batches;
            i += gridDim.x*blockDim.x
        ){
            batch[i] += to_add[i%size];
        }
    }

    int complementBatchBlockSize = 32;
    __global__
    void complementBatch(float* batch, int n, int num_batches)
    {
        for (
            int i = threadIdx.x + blockDim.x*blockIdx.x;
            i < n*n*num_batches;
            i += gridDim.x*blockDim.x
        ){
            int row = (i%(n*n))/n;
            int col = i%(n*n)-row*n;
            batch[i] = ((row==col) ? 1.0f : 0.0f) - batch[i];
        }
    }

    void initBlockSize()
    {
        int min_grid_size;
        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                    &min_grid_size,
                    &addToBatchBlockSize,
                    addToBatch
                ));

        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                    &min_grid_size,
                    &complementBatchBlockSize,
                    complementBatch
                ));
    }
}

Pipe::Pipe(
    int n,
    int m,
    int batch_size,
    FilterParams filter,
    const float* x0,
    const float* P0,
    cublasHandle_t& handle
) : n_(n), m_(m), batch_size_(batch_size), filter_(filter), handle_(handle)
{
    static bool occupancy_calculated = false;
    if (!occupancy_calculated) {
        initBlockSize();
        occupancy_calculated = true;
    }

    checkCudaErrors(cudaMalloc(&vec_1, n_*batch_size_*sizeof(float)));
    checkCudaErrors(cudaMalloc(&vec_2, n_*batch_size_*sizeof(float)));
    checkCudaErrors(cudaMalloc(&vec_3, m_*batch_size_*sizeof(float)));

    int mat_size = std::max(n_*n_, n_*m_);
    checkCudaErrors(cudaMalloc(&mat_1, mat_size*batch_size_*sizeof(float)));
    checkCudaErrors(cudaMalloc(&mat_2, mat_size*batch_size_*sizeof(float)));
    checkCudaErrors(cudaMalloc(&mat_3, mat_size*batch_size_*sizeof(float)));
    checkCudaErrors(cudaMalloc(&mat_4, mat_size*batch_size_*sizeof(float)));

    checkCudaErrors(cudaMalloc(&mat_3_pointers, batch_size_*sizeof(float*)));
    checkCudaErrors(cudaMalloc(&mat_4_pointers, batch_size_*sizeof(float*)));

    for (int i = 0; i < batch_size_; i++) {
        float* a = &mat_3[i*m_*m_];
        float* b = &mat_4[i*m_*m_];
        checkCudaErrors(cudaMemcpy(&mat_3_pointers[i], &a, sizeof(float*), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&mat_4_pointers[i], &b, sizeof(float*), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < batch_size_; i++) {
        checkCudaErrors(cudaMemcpy(&vec_1[i*n_], x0, n_*sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&mat_1[i*n_*n_], P0, n_*n_*sizeof(float), cudaMemcpyHostToDevice));
    }

    info = new int[batch_size_];
}

Pipe::~Pipe()
{
    cudaFree(vec_1);
    cudaFree(vec_2);
    cudaFree(vec_3);

    cudaFree(mat_1);
    cudaFree(mat_2);
    cudaFree(mat_3);
    cudaFree(mat_4);

    cudaFree(mat_3_pointers);
    cudaFree(mat_4_pointers);

    delete[] info;
}

void Pipe::uploadBatch(float* z)
{
    checkCudaErrors(cudaMemcpy(vec_3, z, m_*batch_size_*sizeof(float), cudaMemcpyHostToDevice));
}

void Pipe::downloadBatch(float* x)
{
    checkCudaErrors(cudaMemcpy(x, vec_1, n_*batch_size_*sizeof(float), cudaMemcpyDeviceToHost));
}

void Pipe::xhat()
{
    checkCudaErrors(cublasSgemvStridedBatched(handle_,
        CUBLAS_OP_T,
        n_, n_, &one,
        filter_.F, n_, 0,
        vec_1, 1, n_,
        &zero, vec_2,
        1, n_, batch_size_
    ));

    float* a = vec_1;
    vec_1 = vec_2;
    vec_2 = a;
}

void Pipe::Phat()
{
    // temp = FP
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n_, n_, n_,
        &one,
        mat_1, n_, n_*n_,
        filter_.F, n_, 0,
        &zero, 
        mat_2, n_, n_*n_, 
        batch_size_
    ));

    // P = temp x F^T = FPF^T
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n_, n_, n_,
        &one,
        filter_.F, n_, 0,
        mat_2, n_, n_*n_,
        &zero, 
        mat_1, n_, n_*n_, 
        batch_size_
    ));

    // P += Q
    addToBatch<<<(n_*n_*batch_size_+addToBatchBlockSize-1)/addToBatchBlockSize,addToBatchBlockSize>>>(mat_1, filter_.Q, n_*n_, batch_size_);
}

void Pipe::residual()
{
    checkCudaErrors(cublasSgemvStridedBatched(handle_,
        CUBLAS_OP_T,
        n_, m_, &minus_one,
        filter_.H, n_, 0,
        vec_1, 1, n_,
        &one, vec_3,
        1, m_, batch_size_
    ));
}

void Pipe::gain()
{
    // PH^T
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        m_, n_, n_,
        &one,
        filter_.H, n_, 0,
        mat_1, n_, n_*n_,
        &zero, 
        mat_2, m_, n_*m_, 
        batch_size_
    ));

    // HPH^T
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m_, m_, n_,
        &one,
        mat_2, m_, n_*m_,
        filter_.H, n_, 0,
        &zero, 
        mat_3, m_, m_*m_, 
        batch_size_
    ));

    addToBatch<<<(m_*m_*batch_size_+addToBatchBlockSize-1)/addToBatchBlockSize,addToBatchBlockSize>>>(mat_3, filter_.R, m_*m_, batch_size_);

    // takes array of pointers to other arrays, not strided
    checkCudaErrors(cublasSmatinvBatched(handle_,
        m_, mat_3_pointers, m_, mat_4_pointers, m_, info, batch_size_
    ));

    // PH^T x inv(HPH^T+R)
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m_, n_, m_,
        &one,
        mat_4, m_, m_*m_,
        mat_2, m_, n_*m_,
        &zero, 
        mat_3, m_, n_*m_, 
        batch_size_
    ));
}

void Pipe::updateX()
{
    checkCudaErrors(cublasSgemvStridedBatched(handle_,
        CUBLAS_OP_T,
        m_, n_, &one,
        mat_3, m_, n_*m_,
        vec_3, 1, m_,
        &one, vec_1,
        1, n_, batch_size_
    ));
}

void Pipe::updateP()
{
    // KH
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n_, n_, m_,
        &one,
        filter_.H, n_, 0,
        mat_3, m_, n_*m_,
        &zero, 
        mat_4, n_, n_*n_, 
        batch_size_
    ));

    // I - KH
    complementBatch<<<
        (n_*batch_size_+complementBatchBlockSize-1)/complementBatchBlockSize,
        complementBatchBlockSize
    >>>(mat_4, n_, batch_size_);

    // (I-KH)P
    checkCudaErrors(cublasSgemmStridedBatched(handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n_, n_, n_,
        &one,
        mat_1, n_, n_*n_,
        mat_4, n_, n_*n_,
        &zero, 
        mat_2, n_, n_*n_, 
        batch_size_
    ));

    float* a = mat_1;
    mat_1 = mat_2;
    mat_2 = a;
}
