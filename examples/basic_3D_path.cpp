#include <iostream>
#include <fstream>
#include <sstream>

#include "Matrix.h"
#include "KalmanFilter.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: basic_3D_path <file>" << std::endl;
        return 1;
    }

    DeviceMatrix<float> F(12, 12, {
        1, 1, 1.0/2.0, 1.0/6.0, 0, 0,   0,   0, 0, 0,   0,   0,
        0, 1,   1, 1.0/2.0, 0, 0,   0,   0, 0, 0,   0,   0,
        0, 0,   1,   1, 0, 0,   0,   0, 0, 0,   0,   0,
        0, 0,   0,   1, 0, 0,   0,   0, 0, 0,   0,   0,
        0, 0,   0,   0, 1, 1, 1.0/2.0, 1.0/6.0, 0, 0,   0,   0,
        0, 0,   0,   0, 0, 1,   1, 1.0/2.0, 0, 0,   0,   0,
        0, 0,   0,   0, 0, 0,   1,   1, 0, 0,   0,   0,
        0, 0,   0,   0, 0, 0,   0,   1, 0, 0,   0,   0,
        0, 0,   0,   0, 0, 0,   0,   0, 1, 1, 1.0/2.0, 1.0/6.0,
        0, 0,   0,   0, 0, 0,   0,   0, 0, 1,   1, 1.0/2.0,
        0, 0,   0,   0, 0, 0,   0,   0, 0, 0,   1,   1,
        0, 0,   0,   0, 0, 0,   0,   0, 0, 0,   0,   1
    });

    DeviceMatrix<float> Q(12, 12, {
    1.0/36.0, 1.0/12.0, 1.0/6.0, 1.0/6.0,    0,    0,   0,   0,    0,    0,   0,   0,
    1.0/12.0,  1.0/4.0, 1.0/2.0, 1.0/2.0,    0,    0,   0,   0,    0,    0,   0,   0,
     1.0/6.0,  1.0/2.0,   1,   1,    0,    0,   0,   0,    0,    0,   0,   0,
     1.0/6.0,  1.0/2.0,   1,   1,    0,    0,   0,   0,    0,    0,   0,   0,
       0,    0,   0,   0, 1.0/36.0, 1.0/12.0, 1.0/6.0, 1.0/6.0,    0,    0,   0,   0,
       0,    0,   0,   0, 1.0/12.0,  1.0/4.0, 1.0/2.0, 1.0/2.0,    0,    0,   0,   0,
       0,    0,   0,   0,  1.0/6.0,  1.0/2.0,   1,   1,    0,    0,   0,   0,
       0,    0,   0,   0,  1.0/6.0,  1.0/2.0,   1,   1,    0,    0,   0,   0,
       0,    0,   0,   0,    0,    0,   0,   0, 1.0/36.0, 1.0/12.0, 1.0/6.0, 1.0/6.0,
       0,    0,   0,   0,    0,    0,   0,   0, 1.0/12.0,  1.0/4.0, 1.0/2.0, 1.0/2.0,
       0,    0,   0,   0,    0,    0,   0,   0,  1.0/6.0,  1.0/2.0,   1,   1,
       0,    0,   0,   0,    0,    0,   0,   0,  1.0/6.0,  1.0/2.0,   1,   1
    });

    KalmanFilterState<float> filter_state(
        DeviceMatrix<float>(12, 1, {
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f
        }),
        DeviceMatrix<float>(12, 12, {
            2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f,       0.0f,
                  0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 2000000.0f
        })
    );

    DeviceMatrix<float> R(3, 3, {
        10000.0f,     0.0f,     0.0f,
            0.0f, 10000.0f,     0.0f,
            0.0f,     0.0f, 10000.0f
    });

    DeviceMatrix<float> H(3, 12, {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    });

    KalmanFilter<float> filter(
        std::move(F),
        std::move(Q),
        std::move(R),
        std::move(H)
    );

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cout << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }

    std::vector<DeviceMatrix<float>> zs;

    std::string line;
    std::getline(infile, line); // skip headers
    while(std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        float x = std::stof(tok);
        std::getline(ss, tok, ',');
        float y = std::stof(tok);
        std::getline(ss, tok, ',');
        float z = std::stof(tok);
        zs.push_back(DeviceMatrix<float>(3, 1, {
            x,
            y,
            z
        }));
    }

    std::vector<DeviceMatrix<float>> xs;
    for (const auto& z : zs) {
        filter.run(z, filter_state);
        xs.push_back(filter_state.x);
    }

    std::cout << "x,y,z\n";
    for (const auto& x_dev : xs) {
        HostMatrix<float> x(x_dev);
        std::cout << x(0,0) << "," << x(4,0) << "," << x(8,0) << '\n';
    }

    return 0;
}
