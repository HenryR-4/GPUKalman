#include <iostream>

#include "KalmanFilter.h"

#define TEST(success) if(!success)return 1

int main()
{
    DeviceMatrix<float> F(2, 2, {
        1.0f, 1.0f,
        0.0f, 1.0f
    });
    DeviceMatrix<float> Q(2, 2, {
        25.0f,  0.0f,
         0.0f, 49.0f
    });
    DeviceMatrix<float> x0(2, 1, {
        0.0f,
        0.0f
    });
    DeviceMatrix<float> P0(2, 2, {
        40000.0f,   0.0f,
            0.0f, 400.0f
    });
    DeviceMatrix<float> R(1, 1, {
        10.0f
    });
    DeviceMatrix<float> H(1, 2, {
        1.0f, 0.0f
    });

    HostMatrix<float> Q_host(Q);
 
    KalmanFilter<float> filter(
        std::move(F),
        std::move(Q),
        std::move(x0),
        std::move(P0),
        std::move(R),
        std::move(H)
    );

    filter.run(DeviceMatrix<float>(1, 1, { 53 }));

    return 0;
}
