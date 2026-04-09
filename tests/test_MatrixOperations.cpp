#include <iostream>

#include "Matrix.h"

#define TEST(success) if(!success)return 1

bool test_plusequals()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    A += B;

    HostMatrix<float> expected(2, 2, {
        2.0f, 4.0f,
        6.0f, 8.0f
    });

    HostMatrix<float> result(A);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_plus()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> C = A + B;

    HostMatrix<float> expected(2, 2, {
        2.0f, 4.0f,
        6.0f, 8.0f
    });

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_minusequals()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    A -= B;

    HostMatrix<float> expected(2, 2, {
        0.0f, 0.0f,
        0.0f, 0.0f
    });

    HostMatrix<float> result(A);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_minus()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> C = A - B;

    HostMatrix<float> expected(2, 2, {
        0.0f, 0.0f,
        0.0f, 0.0f
    });

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool testCompoundExpr()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> C = A + B - A;

    HostMatrix<float> expected(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_mulequals()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    A *= B;

    HostMatrix<float> expected(2, 2, {
         7.0f, 10.0f,
        15.0f, 22.0f
    });

    HostMatrix<float> result(A);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_mul()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> B(2, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    DeviceMatrix<float> C = A * B;

    HostMatrix<float> expected(2, 2, {
         7.0f, 10.0f,
        15.0f, 22.0f
    });

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_mul_diff_sizes()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 1.0f,
        0.0f, 1.0f
    });

    DeviceMatrix<float> B(2, 1, {
        0.0f,
        0.0f,
    });

    DeviceMatrix<float> C = A * B;

    HostMatrix<float> expected(2, 1, {
        0.0f,
        0.0f
    });

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_abatpc()
{
    DeviceMatrix<float> A(2, 2, {
        1.0f, 1.0f,
        0.0f, 1.0f
    });

    DeviceMatrix<float> B(2, 2, {
        40000.0f,   0.0f,
            0.0f, 400.0f
    });

    DeviceMatrix<float> C(2, 2, {
        25.0f,  0.0f,
         0.0f, 49.0f
    });

    DeviceMatrix<float> D = abatpc(A, B, C);

    HostMatrix<float> result(D);

    HostMatrix<float> expected(2, 2, {
        40425.0f, 400.0f,
          400.0f, 449.0f
    });

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_abatpc_with_1x1()
{
    DeviceMatrix<float> A(1, 2, {
        1.0f, 0.0f
    });

    DeviceMatrix<float> B(2, 2, {
        40425.0f, 400.0f,
          400.0f, 449.0f
    });

    DeviceMatrix<float> C(1, 1, { 10.0f });

    DeviceMatrix<float> D = abatpc(A, B, C);

    HostMatrix<float> result(D);

    HostMatrix<float> expected(1, 1, { 40435.0f });

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_mul_with_transpose()
{
    DeviceMatrix<float> A(4, 4, {
        200025.0f,      0.0f, 40000.0f,     0.0f,
             0.0f, 200025.0f,     0.0f, 40000.0f,
         40000.0f,      0.0f, 40049.0f,     0.0f,
             0.0f,  40000.0f,     0.0f, 40049.0f
    });

    DeviceMatrix<float> B(2, 4, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    });

    HostMatrix<float> expected(4, 2, {
        200025.0f,      0.0f,
             0.0f, 200025.0f,
         40000.0f,      0.0f,
             0.0f,  40000.0f
    });

    DeviceMatrix<float> C = mul(A, false, B, true);

    HostMatrix<float> result(C);

    if (result != expected) {
        std::cout << "Expected:\n" << expected 
                  << "\nReceived:\n" << result;
        return false;
    }

    std::cout << result;

    return true;
}

bool test_inverse()
{
    DeviceMatrix<float> A(2, 2, {
        210025.0f,      0.0f,
             0.0f, 210025.0f
    });

    A = invert(std::move(A));

    std::cout << HostMatrix<float>(A);

    return true;
}

int main()
{
    std::cout << "\nTEST +=" << '\n';
    TEST(test_plusequals());
    std::cout << "\nTEST +" << '\n';
    TEST(test_plus());

    std::cout << "\nTEST -=" << '\n';
    TEST(test_minusequals());
    std::cout << "\nTEST -" << '\n';
    TEST(test_minus());

    std::cout << "\nTEST compound expression" << '\n';
    TEST(testCompoundExpr());

    std::cout << "\nTEST *=" << '\n';
    TEST(test_mulequals());
    std::cout << "\nTEST *" << '\n';
    TEST(test_mul());
    std::cout << "\nTEST * with different sized matricies" << '\n';
    TEST(test_mul_diff_sizes());

    std::cout << "\nTEST ABA^T + C" << '\n';
    TEST(test_abatpc());

    std::cout << "\nTEST ABA^T + C with 1x1" << '\n';
    TEST(test_abatpc_with_1x1());

    std::cout << "\nTEST A*B_T" << '\n';
    TEST(test_mul_with_transpose());

    std::cout << "\nTEST A^-1" << '\n';
    TEST(test_inverse());
    return 0;
}
