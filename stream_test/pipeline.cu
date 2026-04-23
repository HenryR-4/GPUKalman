#include <vector>

const int num_pipes = 2;
const int num_batches = 5;
const int num_elements = 25000000; 

__device__
void sleep(int ms)
{
    for (int i = 0; i < ms; i++) {
        __nanosleep(1000000U);
    }
}
__global__ void xhat() { sleep(10); }
__global__ void Phat() { sleep(15); }
__global__ void y() { sleep(10); }
__global__ void K() { sleep(25); }
__global__ void x() { sleep(15); }
__global__ void P() {sleep(20); }

struct Pipe
{
    Pipe()
    {
        cudaStreamCreate(&s1);
        cudaStreamCreate(&s2);
        cudaEventCreate(&e);
    }

    ~Pipe()
    {
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
        cudaEventDestroy(e);
    }

    cudaStream_t s1;
    cudaStream_t s2;
    cudaEvent_t e;
};

int main()
{
    cudaStream_t uploadStream;
    cudaStreamCreate(&uploadStream);

    size_t bytes = num_elements*sizeof(float); 

    float* data_host;
    cudaMallocHost(&data_host, bytes);

    float* data;
    cudaMalloc(&data, bytes);

    std::vector<Pipe> pipes(num_pipes);

    for (int i = 0; i < num_batches; i++) {
        Pipe& pipe = pipes[i%pipes.size()];

        // upload batch
        cudaMemcpyAsync(data, data_host, bytes, cudaMemcpyHostToDevice, uploadStream);
        cudaEventRecord(pipe.e, uploadStream);

        // predict
        xhat<<<1,1,0,pipe.s1>>>();
        Phat<<<1,1,0,pipe.s2>>>();

        // residual and Kalman gain
        cudaStreamWaitEvent(pipe.s1, pipe.e, 0);
        cudaStreamWaitEvent(pipe.s2, pipe.e, 0);
        y<<<1,1,0,pipe.s1>>>();
        K<<<1,1,0,pipe.s2>>>();
        cudaEventRecord(pipe.e, pipe.s2);

        // update
        cudaStreamWaitEvent(pipe.s1, pipe.e, 0);
        x<<<1,1,0,pipe.s1>>>();
        P<<<1,1,0,pipe.s2>>>();
        cudaEventRecord(pipe.e, pipe.s2);

        // download batch
        cudaStreamWaitEvent(pipe.s1, pipe.e, 0);
        cudaMemcpyAsync(data_host, data, bytes, cudaMemcpyDeviceToHost, pipe.s1);

        // synchronize so predict stage does not overload download
        cudaEventRecord(pipe.e, pipe.s1);
        cudaStreamWaitEvent(pipe.s2, pipe.e, 0);
    }

    cudaFree(data);
    cudaFreeHost(data_host);

    cudaStreamDestroy(uploadStream);

    return 0;
}
