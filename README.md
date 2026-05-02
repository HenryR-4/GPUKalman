# GPU Kalman Filtering

This project is about tracking as many states as possible using the parallelization offered by GPU hardware.
It is written in CUDA primarily leveraging the cuBLAS library.

This is not a complete usable library, more work would be required to make it so.
As such, the code is written with only a single basic filter design, although it should be relatively easy to create any arbitrary filter.
It is kept basic as the focus is on the GPU implementation not filter design.

The filter used is a simple 2D position tracker where the measurement input is a noisy (x,y) position vector and the state is a (x, vx, y, vy) position and velocity vector.

There are two implementations, one which executes entirely on the default stream and another which leverages multiple streams. 
Both use batched kernels since typically the filter matricies and states are quite small by themselves.

## Compiling

The code consists of two projects: Batched, BatchedWithStreams.
Each are built separately. 

You can compile them using cmake.
You will need to set the correct CUDA definitions.

If running on asax you can simply run the asax_build.sh script which will compile both projects.
```
./scripts/asax_build.sh
```

### Dependencies
- CUDAToolkit: cudart, cublas

## Running
On asax you can run:
```
./scripts/submitall.sh
```
This will submit all the tests for timing results which will be placed in the output directory.
But be warned it will submit over 150 jobs.
If you do decide to try this, there is a glob_results.sh script which will gather all the times and print them in a csv format.

This script calls the runBatched.pbs and runBatchedWithStreams.pbs scripts internally.
You can look at the submitall.sh script to see how to use them.

The tests can also be run like this in your own pbs script:
```
./build/Batched/test <input file> [-p]
./build/BatchedWithStreams/test <input file> <num pipes> [-p]
```
-p indicates that the filter output should be printed in csv format, redirect the output to a file.
If you want to use the plot script you must remove the last line at the bottom of the output file that contains the timing result.

## Generating Data

There is one example data file provided with 16 noisy paths if you do not want to generate more test data.
I have excluded more due to file size.

If you run the script it should tell you the arguments that it takes.
Typically you will run the generate2D script followed by the basic_noisy_measurement script.
All outputs should be redirected to a file.

The filter was designed with a variance of 100 provided to the basic_noisy_measurement script.
I would recommend to use this value.

### Script Dependencies 
- python3, matplotlib, numpy 
