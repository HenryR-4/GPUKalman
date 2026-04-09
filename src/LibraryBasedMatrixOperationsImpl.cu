#include "Matrix.h"

#include "cublas_v2.h"
#include "cusolverDn.h"

struct LibraryState
{
    LibraryState();
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
};

LibraryState::LibraryState()
{
    cublasCreate(&cublasH);
    cusolverDnCreate(&cusolverH);
}

LibraryState libstate;

template<>
DeviceMatrix<float>& operator+=(DeviceMatrix<float>& lhs, const DeviceMatrix<float>& rhs)
{
    float a = 1.0f;
    float b = 1.0f;

    cublasSgeam(libstate.cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        lhs.width(), rhs.height(),
        &a, lhs.data(), lhs.width(),
        &b, rhs.data(), rhs.width(),
        lhs.data(), lhs.width()
    );

    return lhs;
}

template<>
DeviceMatrix<float> operator+(DeviceMatrix<float> lhs, const DeviceMatrix<float>& rhs)
{
    lhs += rhs;
    return lhs;
}

template<>
DeviceMatrix<float>& operator-=(DeviceMatrix<float>& lhs, const DeviceMatrix<float>& rhs)
{
    float a = 1.0f;
    float b = -1.0f;

    cublasSgeam(libstate.cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        lhs.width(), rhs.height(),
        &a, lhs.data(), lhs.width(),
        &b, rhs.data(), rhs.width(),
        lhs.data(), lhs.width()
    );

    return lhs;
}

template<>
DeviceMatrix<float> operator-(DeviceMatrix<float> lhs, const DeviceMatrix<float>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template<>
bool operator!=(const HostMatrix<float>& lhs, const HostMatrix<float>& rhs)
{
    if (lhs.height() != rhs.height() || lhs.width() != rhs.width()) {
        return true;
    }

    for (int i = 0; i < lhs.height(); i++) {
        for (int j = 0; j < lhs.width(); j++) {
            if (lhs(i,j) != rhs(i,j)) {
                return true;
            }
        }
    }

    return false;
}

template<>
DeviceMatrix<float>& operator*=(DeviceMatrix<float>& lhs, const DeviceMatrix<float>& rhs)
{
    float a = 1.0f;
    float b = 0.0f;

    DeviceMatrix<float> temp(lhs.height(), rhs.width());
    cublasSgemm(libstate.cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        rhs.width(), lhs.height(), lhs.width(),
        &a, rhs.data(), rhs.width(),
        lhs.data(), lhs.width(),
        &b,
        temp.data(), rhs.width()
    );

    lhs = std::move(temp);

    return lhs;
}

template<>
DeviceMatrix<float> operator*(DeviceMatrix<float> lhs, const DeviceMatrix<float>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template<>
DeviceMatrix<float> abatpc(
    const DeviceMatrix<float>& A,
    const DeviceMatrix<float>& B,
    const DeviceMatrix<float>& C
){
    DeviceMatrix<float> AB = A*B;

    float a = 1.0f;
    float b = 1.0f;

    DeviceMatrix<float> temp(C);
    cublasSgemm(libstate.cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        A.height(), AB.height(), AB.width(),
        &a, A.data(), A.width(),
        AB.data(), AB.width(),
        &b,
        temp.data(), A.height()
    );

    return temp;
}

template<>
DeviceMatrix<float> mul(
    const DeviceMatrix<float>& A,
    bool transA,
    const DeviceMatrix<float>& B,
    bool transB
){
    float a = 1.0f;
    float b = 0.0f;

    float m = transB ? B.height() : B.width();
    float n = transA ? A.width() : A.height();
    float k = transA ? A.height() : A.width();

    DeviceMatrix<float> temp(n, m);
    cublasSgemm(libstate.cublasH, 
        transB ? CUBLAS_OP_T : CUBLAS_OP_N, 
        transA ? CUBLAS_OP_T : CUBLAS_OP_N, 
        m, n, k,
        &a, B.data(), B.width(),
        A.data(), A.width(),
        &b,
        temp.data(), m
    );

    return temp;
}

template<>
DeviceMatrix<float> invert(DeviceMatrix<float>&& A)
{
    int lwork;
    float* d_work;
    int* d_info;

    cusolverDnSgetrf_bufferSize(libstate.cusolverH,
        A.height(), A.width(), A.data(), A.width(),
        &lwork
    );

    cudaMalloc(&d_work, sizeof(float)*lwork);
    cudaMalloc(&d_info, sizeof(int));

    // LU factorization
    cusolverDnSgetrf(libstate.cusolverH,
        A.width(),
        A.height(),
        A.data(),
        A.width(),
        d_work,
        nullptr,
        d_info
    );

    HostMatrix<float> B_host(A.height(), A.width());
    for (size_t i = 0; i < B_host.height(); i++) {
        for (size_t j = 0; j < B_host.width(); j++) {
            B_host(i,j) = (i == j) ? 1.0f : 0.0f;
        }
    }

    DeviceMatrix<float> B(B_host);
    cusolverDnSgetrs(libstate.cusolverH,
        CUBLAS_OP_N, A.height(), B.height(),
        A.data(), A.width(), nullptr,
        B.data(), B.width(), d_info
    );

    cudaFree(d_work);
    cudaFree(d_info);

    return B;
}
