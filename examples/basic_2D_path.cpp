#include <iostream>
#include <fstream>
#include <sstream>

#include "Matrix.h"
#include "KalmanFilter.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: basic_2D_path <file>" << std::endl;
        return 1;
    }

    DeviceMatrix<float> F(4, 4, {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    });
    DeviceMatrix<float> Q(4, 4, {
        25.0f,  0.0f,  0.0f,  0.0f,
         0.0f, 25.0f,  0.0f,  0.0f,
         0.0f,  0.0f, 49.0f,  0.0f,
         0.0f,  0.0f,  0.0f, 49.0f,
    });
    KalmanFilterState<float> filter_state({
        DeviceMatrix<float>(4, 1, {
            0.0f,
            0.0f,
            0.0f,
            0.0f
        }),
        DeviceMatrix<float>(4, 4, {
            160000.0f,      0.0f,     0.0f,     0.0f,
                 0.0f, 160000.0f,     0.0f,     0.0f,
                 0.0f,      0.0f, 40000.0f,     0.0f,
                 0.0f,      0.0f,     0.0f, 40000.0f
        })
    });
    DeviceMatrix<float> R(2, 2, {
        10000.0f,     0.0f,
            0.0f, 10000.0f
    });
    DeviceMatrix<float> H(2, 4, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
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
        zs.push_back(DeviceMatrix<float>(2, 1, {
            x,
            y
        }));
    }

    std::vector<HostMatrix<float>> xs;
    for (const auto& z : zs) {
        filter.run(z, filter_state);
        xs.push_back(HostMatrix<float>(filter_state.x));
    }

    std::cout << "x,y\n";
    for (const auto& x : xs) {
        std::cout << x(0,0) << "," << x(1,0) << '\n';
    }

    return 0;
}
