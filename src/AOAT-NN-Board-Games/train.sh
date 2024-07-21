#!/bin/bash
for i in {1..45}
do
    python3 Simulate.py $i
    python3 Learn.py $i
done