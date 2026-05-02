#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

#include <helper_cuda.h>

#include "pipe.h"

#define N 4
#define M 2

float* F;
float* Q;
float* H;
float* R;

void init()
{
    const float F_host[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    checkCudaErrors(cudaMalloc(&F, N*N*sizeof(float)));
    checkCudaErrors(cudaMemcpy(F, F_host, N*N*sizeof(float), cudaMemcpyHostToDevice));

    const float Q_host[] = {
        25.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 49.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 25.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 49.0f,
    };
    checkCudaErrors(cudaMalloc(&Q, N*N*sizeof(float)));
    checkCudaErrors(cudaMemcpy(Q, Q_host, N*N*sizeof(float), cudaMemcpyHostToDevice));

    const float H_host[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    checkCudaErrors(cudaMalloc(&H, N*M*sizeof(float)));
    checkCudaErrors(cudaMemcpy(H, H_host, N*M*sizeof(float), cudaMemcpyHostToDevice));

    const float R_host[] = {
        10000.0f, 0.0f,
        0.0f, 10000.0f
    };
    checkCudaErrors(cudaMalloc(&R, M*M*sizeof(float)));
    checkCudaErrors(cudaMemcpy(R, R_host, M*M*sizeof(float), cudaMemcpyHostToDevice));
}

void cleanup()
{
    cudaFree(F);
    cudaFree(Q);
    cudaFree(H);
    cudaFree(R);
}

std::pair<std::vector<float>,int> read_measurements(const std::string& filename)
{
    std::vector<float> measurements;

    std::ifstream file(filename);

    std::string line;

    // use headers to count
    std::getline(file, line);
    int num_measurements = 0;
    {
        int i = 0;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok,',')) {
            i++;
            if (i%M == 0) { num_measurements++; }
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string tok;
        while(std::getline(ss, tok, ',')) {
            measurements.push_back(std::stof(tok));
        }
    }

    return {measurements, num_measurements};
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cout << "Usage: ./program <input file> <# pipes> [-p]\n";
        return 1;
    }
    bool print = false;
    if (argc > 3){
        std::string arg = argv[3];
        if (arg == "-p") { print = true; }
    }

    int num_pipes;
    try {
        num_pipes = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cout << "Invalid # pipes: " << argv[2] << "\n";
        std::cout << "Usage: ./program <input file> <# pipes> [-p]\n";
        std::cout << e.what() << std::endl;
        return 1;
    }

    std::pair<std::vector<float>,int> zs;
    try {
         zs = read_measurements(argv[1]);
    } catch (const std::exception& e) {
        std::cout << "Failed to read measurements from: " << argv[1] << "\n";
        std::cout << e.what() << std::endl;
    }

    if (zs.first.size() < 1) {
        std::cout << "Unable to load measurements from: " << argv[1] << "\n";
        return 1;
    }

    int batch_size = zs.second;
    if (batch_size < 1) {
        std::cout << "There must be at least 1 set of measurements in file\n";
        return 1;
    }

    init();

    const float x0[] = {0.0f,0.0f,0.0f,0.0f};
    const float P0[] = {
        160000.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 40000.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 160000.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 40000.0f,
    };
    FilterParams filter = { F, Q, H, R };

    std::vector<std::unique_ptr<Pipe>> pipes(num_pipes);
    int sizes[num_pipes];
    for (int i = 0; i < num_pipes; i++) {
        sizes[i] = batch_size/num_pipes;
        if (i < batch_size%num_pipes) { sizes[i]++; }
        pipes[i] = std::make_unique<Pipe>(N, M, sizes[i], filter, x0, P0);
    }

    const int num_times = zs.first.size()/(M*batch_size);

    float* measurements;
    checkCudaErrors(cudaMallocHost(&measurements,zs.first.size()*sizeof(float)));
    checkCudaErrors(cudaMemcpy(measurements, zs.first.data(), zs.first.size()*sizeof(float), cudaMemcpyHostToHost));

    float* results;
    checkCudaErrors(cudaMallocHost(&results,batch_size*N*num_times*sizeof(float)));

    float* current_result = results;
    float* current_measurement = measurements;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_pipes; i++) {
        pipes[i]->uploadFirstBatch(current_measurement);
        current_measurement += M*sizes[i];
    }
    for (int i = 0; i < num_times; i++) {
        for (int j = 0; j < num_pipes; j++) {
            pipes[j]->xhat();
            pipes[j]->Phat();

            pipes[j]->residual();
            pipes[j]->gain();

            pipes[j]->updateX();
            pipes[j]->updateP();

            // upload measurement for NEXT! batch
            // no next batch on last time step
            if (i < num_times-1) {
                pipes[j]->uploadBatch(current_measurement);
                current_measurement += M*sizes[j];
            }

            pipes[j]->downloadBatch(current_result);
            current_result += N*sizes[j];
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    if (print) {
        for (int i = 0; i < num_times; i++) {
            for (int j = 0; j < batch_size; j++) {
                std::cout << results[i*batch_size*N+j*N] << "," << results[i*batch_size*N+j*N+2]; 
                if (j < batch_size-1) { std::cout << ","; }
            }
            std::cout << "\n";
        }
    }
    std::cout << "Avg Time Per Step: " << time / num_times << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cleanup();

    return 0;
}
