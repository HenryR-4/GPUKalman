#include <iostream>
#include <fstream>
#include <sstream>

#include "Matrix.h"
#include "KalmanFilter.h"
#include "DeviceTimer.h"

// return array of arrays of measurements
// i.e. size of the return is the number of paths
std::vector<std::vector<DeviceMatrix<float>>> loadMeasurements(const std::string& filename)
{
    std::vector<std::vector<DeviceMatrix<float>>> result;

    std::ifstream infile(filename);

    std::string line;
    int dim = 3;
    std::getline(infile, line); // skip header
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<float> coord(3);
        int i = 0;
        int j = 0;
        while (std::getline(ss, tok, ',')) {
            coord[i] = std::stof(tok);
            i++;
            if (i == 3) {
                i = 0;
                if (j >= result.size()) { result.push_back({}); }
                result[j].push_back(DeviceMatrix<float>(3,1, coord));
                j++;
            }
        }
    }

    return result;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: basic_3D_path <file>" << std::endl;
        return 1;
    }

    std::cerr << "Loading..." << std::endl;
    std::vector<std::vector<DeviceMatrix<float>>> zs;
    try {
         zs = loadMeasurements(argv[1]); 
    } catch (...) {
        std::cout << "Error: Failed to read file" << std::endl;
        std::cout << "    Usage: basic_3D_path <file>" << std::endl;
        return 1;
    }

    // 1 filter state per
    KalmanFilterState<float> initial_state(
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
    std::vector<KalmanFilterState<float>> filter_states(zs.size(), initial_state); 

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

    DeviceTimer t;

    std::cerr << "Timing..." << std::endl;
    t.start();
    std::vector<std::vector<DeviceMatrix<float>>> xs(zs.size());
    for (size_t i = 0; i < zs.size(); i++) {
        for (const auto& z : zs[i]) {
            filter.run(z, filter_states[i]);
            // blocking dev-dev memcpy between each iteration :(
            xs[i].push_back(filter_states[i].x); 
        }
    }
    t.stop();
    std::cerr << t.get_elapsed_time_ms() << std::endl;

    std::vector<std::vector<HostMatrix<float>>> results(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        for (const auto& x : xs[i]) {
            results[i].push_back(HostMatrix<float>(x));
        }
    }

    for (size_t i = 0; i < results[0].size(); i++) {
        if (i == 0) {
            for (size_t j = 0; j < results.size(); j++) {
                std::cout << "x" << j << ",y" << j << ",z" << j;
                if (j != results.size()-1) { std::cout << ","; }
            }
            std::cout << "\n";
        }
 
        for (size_t j = 0; j < results.size(); j++) {
            std::cout << results[j][i](0,0) << "," << results[j][i](4,0) << "," << results[j][i](8,0);
            if (j != results.size()-1) { std::cout << ","; }
        }
        std::cout << "\n";
    }

    return 0;
}
