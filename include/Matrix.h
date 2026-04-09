#pragma once

#include <vector>
#include <stdexcept>
#include <cassert>
#include <ostream>
#include "cuda_runtime.h"

template<typename T> 
class Matrix
{
    public:
        Matrix(): data_(nullptr) {}
        Matrix(size_t height, size_t width) 
            : height_(height),
              width_(width)
        {
            size_ = height_*width_;
            bytes_ = size_*sizeof(T);
        }

        T* data() { return data_; }
        const T* data() const { return data_; }
        size_t height() const { return height_; }
        size_t width() const { return width_; }
        size_t size() const { return size_; }
        size_t bytes() const { return bytes_; }

    protected:
        T* data_;
        size_t height_;
        size_t width_;
        size_t size_;
        size_t bytes_;
};

template<typename T> class DeviceMatrix;

template<typename T>
class HostMatrix : public Matrix<T>
{
    public:
        HostMatrix(const DeviceMatrix<T>& device_matrix);
        HostMatrix(HostMatrix<T>&& other);
        HostMatrix(size_t height, size_t width);
        HostMatrix(size_t height, size_t width, const std::vector<T>& host_matrix);
        ~HostMatrix();
        T& operator()(size_t row, size_t col);
        const T& operator()(size_t row, size_t col) const;
};

template<typename T>
class DeviceMatrix : public Matrix<T>
{
    static_assert(std::is_same<T,float>::value, "Only 'float' supported.");
    public:
        DeviceMatrix() = default;
        DeviceMatrix(size_t height, size_t width);
        DeviceMatrix(size_t height, size_t width, const std::vector<T>& host_matrix);
        DeviceMatrix(const HostMatrix<T>& host_matrix);
        DeviceMatrix(const DeviceMatrix<T>& other);
        DeviceMatrix(DeviceMatrix<T>&& other);
        ~DeviceMatrix();

        DeviceMatrix<T>& operator=(const DeviceMatrix<T>& other);
        DeviceMatrix<T>& operator=(DeviceMatrix<T>&& other);

        struct DeviceMatrixView;
        DeviceMatrixView view();
    private:
        void allocate();
};

template<typename T>
HostMatrix<T>::HostMatrix(const DeviceMatrix<T>& device_matrix)
    : Matrix<T>(device_matrix.height(), device_matrix.width())
{
    this->data_ = new T[this->size_];
    cudaError_t err = cudaMemcpy(this->data_, device_matrix.data(), this->bytes_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
HostMatrix<T>::HostMatrix(HostMatrix<T>&& other)
    : Matrix<T>(other.height_, other.width_)
{
    this->data_ = other.data_;
    other.data_ = nullptr;
}

template<typename T>
HostMatrix<T>::HostMatrix(size_t height, size_t width, const std::vector<T>& host_matrix)
    : Matrix<T>(height, width)
{
    assert(host_matrix.size() == height*width);
    this->data_ = new T[this->size_];
    std::copy(host_matrix.begin(), host_matrix.end(), this->data_);
}

template<typename T>
HostMatrix<T>::HostMatrix(size_t height, size_t width)
    : Matrix<T>(height, width)
{
    this->data_ = new T[this->size_];
}

template<typename T>
HostMatrix<T>::~HostMatrix()
{
    delete[] this->data_;
}

template<typename T>
T& HostMatrix<T>::operator()(size_t row, size_t col)
{
    return this->data_[row*this->width_+col];
}

template<typename T>
const T& HostMatrix<T>::operator()(size_t row, size_t col) const
{
    return this->data_[row*this->width_+col];
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(size_t height, size_t width)
    : Matrix<T>(height, width)
{
    allocate();
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(size_t height, size_t width, const std::vector<T>& host_matrix)
    : Matrix<T>(height, width)
{
    assert(host_matrix.size() == height*width);
    allocate();

    cudaError_t err = cudaMemcpy(this->data_, host_matrix.data(), this->bytes_, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(const HostMatrix<T>& host_matrix)
    : Matrix<T>(host_matrix.height(), host_matrix.width())
{
    allocate();
    cudaError_t err = cudaMemcpy(this->data_, host_matrix.data(), this->bytes_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(const DeviceMatrix<T>& other)
    : Matrix<T>(other.height_, other.width_)
{
    allocate();
    cudaError_t err = cudaMemcpy(this->data_, other.data_, this->bytes_, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(DeviceMatrix<T>&& other)
    : Matrix<T>(other.height_, other.width_)
{
    this->data_ = other.data_;
    other.data_ = nullptr;
}

template<typename T>
DeviceMatrix<T>& DeviceMatrix<T>::operator=(const DeviceMatrix<T>& other)
{
    if (this->data_ != nullptr) {
        cudaError_t err = cudaFree(this->data_);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    this->height_ = other.height_;
    this->width_ = other.width_;
    this->size_ = other.size_;
    this->bytes_ = other.bytes_;
    this->data_ = other.data_;
    allocate();
    cudaError_t err = cudaMemcpy(this->data_, other.data_, this->bytes_, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return *this;
}

template<typename T>
DeviceMatrix<T>& DeviceMatrix<T>::operator=(DeviceMatrix<T>&& other)
{
    if (this->data_ != nullptr) {
        cudaError_t err = cudaFree(this->data_);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    this->height_ = other.height_;
    this->width_ = other.width_;
    this->size_ = other.size_;
    this->bytes_ = other.bytes_;
    this->data_ = other.data_;
    other.data_ = nullptr;

    return *this;
}

template<typename T>
void DeviceMatrix<T>::allocate()
{
    cudaError_t err = cudaMalloc(&this->data_, this->bytes_);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
DeviceMatrix<T>::~DeviceMatrix()
{
    cudaFree(this->data_);
}

template<typename T>
struct DeviceMatrix<T>::DeviceMatrixView
{
    T* data;
    size_t height;
    size_t width;

    __device__ T& operator()(size_t row, size_t col) 
    {
        return data[row*width+col];
    }
};

template<typename T>
struct DeviceMatrix<T>::DeviceMatrixView DeviceMatrix<T>::view()
{
    return { this->data_, this->height_, this->width_ };
}

// OPERATORS ==========================
template<typename T>
DeviceMatrix<T>& operator+=(DeviceMatrix<T>& lhs, const DeviceMatrix<T>& rhs);
template<typename T>
DeviceMatrix<T> operator+(DeviceMatrix<T> lhs, const DeviceMatrix<T>& rhs);

template<typename T>
DeviceMatrix<T>& operator-=(DeviceMatrix<T>& lhs, const DeviceMatrix<T>& rhs);
template<typename T>
DeviceMatrix<T> operator-(DeviceMatrix<T> lhs, const DeviceMatrix<T>& rhs);

template<typename T>
DeviceMatrix<T>& operator*=(DeviceMatrix<T>& lhs, const DeviceMatrix<T>& rhs);
template<typename T>
DeviceMatrix<T> operator*(DeviceMatrix<T> lhs, const DeviceMatrix<T>& rhs);

// ABA^T + C
template<typename T>
DeviceMatrix<T> abatpc(
    const DeviceMatrix<T>& A,
    const DeviceMatrix<T>& B,
    const DeviceMatrix<T>& C
);

template<typename T>
DeviceMatrix<T> mul(
    const DeviceMatrix<T>& A,
    bool transA,
    const DeviceMatrix<T>& B,
    bool transB
);

template<typename T>
DeviceMatrix<T> invert(DeviceMatrix<T>&& A);

template<typename T>
bool operator!=(const HostMatrix<T>& lhs, const HostMatrix<T>& rhs);

template<typename T>
std::ostream& operator<<(std::ostream& os, const HostMatrix<T>& A)
{
    for (size_t i = 0; i < A.height(); i++) {
        size_t j;
        os << "[ ";
        for (j = 0; j < A.width()-1; j++) {
            os << A(i,j) << ", ";
        }
        os << A(i,j) << " ]\n";
    }
    return os;
}
