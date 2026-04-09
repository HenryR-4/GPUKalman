#include <DeviceTimer.h>

#include <cuda_runtime.h>
#include <helpers/helper_cuda.h>

#include <chrono>

class DeviceTimer::Implementation {
public:
  // Rule of Five
  Implementation() {
    checkCudaErrors(cudaEventCreate(&start_));
    checkCudaErrors(cudaEventCreate(&stop_));
  }

  ~Implementation() {
    checkCudaErrors(cudaEventDestroy(start_));
    checkCudaErrors(cudaEventDestroy(stop_));
  }

  Implementation(Implementation const &) = delete;
  Implementation(Implementation &&) = delete;
  
  Implementation & operator=(Implementation const &) = delete;
  Implementation & operator=(Implementation &&) = delete;
  // end Rule of Five

  void start() {
    checkCudaErrors(cudaEventRecord(start_));
  }

  void stop() {
    checkCudaErrors(cudaEventRecord(stop_));
  }

  float get_elapsed_time_ms() {
    checkCudaErrors(cudaEventSynchronize(stop_));
    float result = 0.0;
    checkCudaErrors(cudaEventElapsedTime(&result, start_, stop_));
    return result;
  }

private:
  
  cudaEvent_t start_;
  cudaEvent_t stop_;

};

DeviceTimer::DeviceTimer() : implementation_(std::make_shared<Implementation>()) {}

void DeviceTimer::start() { implementation_->start(); }

void DeviceTimer::stop() { implementation_->stop(); }

float DeviceTimer::get_elapsed_time_ms() {
  return implementation_->get_elapsed_time_ms();
}
