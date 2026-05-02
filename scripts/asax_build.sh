#!/usr/bin/env bash

set -e

CUDA_HOME=/apps/x86-64/apps/cuda_12.6.0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BATCH_DIR=$SCRIPT_DIR/../Batched
STREAM_DIR=$SCRIPT_DIR/../BatchedWithStreams

BUILD_DIR=$SCRIPT_DIR/../build/
BATCH_BUILD_DIR=$BUILD_DIR/Batched
STREAM_BUILD_DIR=$BUILD_DIR/BatchedWithStreams

rm -rf $BUILD_DIR
mkdir -p $BATCH_BUILD_DIR
mkdir -p $STREAM_BUILD_DIR

cmake \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -S $BATCH_DIR \
  -B $BATCH_BUILD_DIR

cmake --build $BATCH_BUILD_DIR

cmake \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -S $STREAM_DIR \
  -B $STREAM_BUILD_DIR

cmake --build $STREAM_BUILD_DIR
