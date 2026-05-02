#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BUILD_DIR=$SCRIPT_DIR/../build/
BATCH_BUILD_DIR=$BUILD_DIR/Batched
STREAM_BUILD_DIR=$BUILD_DIR/BatchedWithStreams
OUTPUT_DIR=$SCRIPT_DIR/../output
INPUT_DIR=$SCRIPT_DIR/../input

if [ ! -d "$INPUT_DIR" ]; then
	echo "Missing Input Dir: $INPUT_DIR"
	echo "Use provided scripts to generate test data"
	exit 1
fi

mkdir -p $OUTPUT_DIR

qsub -N example_batch -v "PROGRAM=$BATCH_BUILD_DIR/test,N=16,OUTPUT_DIR=$OUTPUT_DIR,INPUT_DIR=$INPUT_DIR,PRINT=1" $SCRIPT_DIR/runBatched.pbs
qsub -N example_stream -v "PROGRAM=$STREAM_BUILD_DIR/test,N=16,OUTPUT_DIR=$OUTPUT_DIR,INPUT_DIR=$INPUT_DIR,PIPES=1,PRINT=1" $SCRIPT_DIR/runBatchedWithStreams.pbs
