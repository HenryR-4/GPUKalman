#!/usr/bin/env bash

echo -n 'num_states,batched,'
for i in {0..7}; do
	echo -n "streams_$((2**i))"
	if [ $i -ne 7 ]; then
		echo -n ','
	fi
done
echo ''

for i in {0..16}; do
	paste -d, <(echo "$((2**i))") \
	<(cat output/test-batched-$((2**i)).out | awk '{print $5}') \
	<(for j in {0..7}; do cat output/test-$((2**j))-streams-$((2**i)).out | if [ $j -ne 7 ]; then awk '{printf "%s,", $5}'; else awk '{printf "%s", $5}'; fi done)
done
