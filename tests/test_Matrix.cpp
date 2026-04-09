#include <iostream>
#include <sstream>
#include "Matrix.h"

bool deviceToHost()
{
    DeviceMatrix<float> A_device(2, 2, {
		1.0f, 2.0f,
		3.0f, 4.0f
    });

    HostMatrix<float> A_host(A_device);

    std::stringstream ss;
    ss << A_host;

    std::string expected = 
        "[ 1, 2 ]\n"
        "[ 3, 4 ]\n";

    std::cout << ss.str();

    if (ss.str() != expected) {
        return false;
    }

    return true;
}

bool hostToDevice()
{
    HostMatrix<float> A_host(2, 2, {
		1.0f, 2.0f,
		3.0f, 4.0f
    });

    DeviceMatrix<float> A_device(A_host);
    HostMatrix<float> res(A_device);

    std::stringstream ss;
    ss << res;

    std::string expected = 
        "[ 1, 2 ]\n"
        "[ 3, 4 ]\n";

    std::cout << ss.str();

    if (ss.str() != expected) {
        return false;
    }

    return true;
}

bool copyAssign()
{
    DeviceMatrix<float> A;
    DeviceMatrix<float> B(2, 2, {
		1.0f, 2.0f,
		3.0f, 4.0f
    });

    float* addr = B.data();

    A = B;

    if (A.data() == addr) {
        return false;
    }

    return true;
}

bool moveAssign()
{
    DeviceMatrix<float> A;
    DeviceMatrix<float> B(2, 2, {
		1.0f, 2.0f,
		3.0f, 4.0f
    });

    float* addr = B.data();

    A = std::move(B);

    if (A.data() != addr) {
        return false;
    }

    return true;
}

bool deviceToDevice()
{
    DeviceMatrix<float> A(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    float* addrA = A.data();
    DeviceMatrix<float> B(A);
    float* addrB = B.data();
    if (addrA == addrB || addrA == nullptr || addrB == nullptr) {
        std::cout << "Device Matrix Copy Constructor Failed" << std::endl;
        return false;
    }

    DeviceMatrix<float> C(std::move(A));
    float* addrC = C.data();
    if (addrC != addrA || A.data() != nullptr) {
        std::cout << "Device Matrix Move Constructor Failed" << std::endl;
        return false;
    }

    return true;
}


#define TEST(success) if(!success)return 1

int main()
{
    TEST(deviceToHost());
    TEST(hostToDevice());
    TEST(deviceToDevice());
    TEST(copyAssign());
    TEST(moveAssign());
}
