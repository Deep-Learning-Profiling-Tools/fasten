#!/bin/bash

phase=("forward" "backward")
slices=("AIFB" "AM" "BGS" "MUTAG")
code=("fasten" "pyg" "torch")
K=(32 64 128)

cd test || exit

# get the length of the slices array
slices_len=${#slices[@]}
output="output.log"
rm -rf $output
touch $output

rm -rf perf.*

for c in "${code[@]}"
do
  for p in "${phase[@]}"
  do
      for s_i in $(seq 0 "$slices_len")
      do
          s=${slices[$s_i]}
          for k in "${K[@]}"
          do
              options=$k-$s-slices$s_i-$c-float32-$p
              nsys profile -o perf -f true pytest -vs test_ops.py::test_perf["$options"]
              nsys stats --report cuda_gpu_kern_sum perf.nsys-rep -f csv -o 1
              echo "$options" >> $output
              grep segment_matmul_kernel 1_cuda_gpu_kern_sum.csv | cut -d "," -f 6 >> $output
              tail -n 2 $output
              rm 1_cuda_gpu_kern_sum.csv
              rm -rf perf.*
          done
      done
  done
done

cd .. || exit
