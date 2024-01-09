#!/bin/bash

# Check number of arguments
mode="dataset"

if [ "$#" -eq 1 ]; then
    mode=$1
fi

echo "Mode: $mode"

phase=("forward" "backward")
slices=("AIFB" "AM" "BGS" "MUTAG")
code=("fasten" "pyg")

K=(32 64 128)
if [ "$mode" == "random" ]; then
    K=(32 128)
fi

cd test || exit

# get the length of the slices array
slices_len=${#slices[@]}
output="output.log"
rm -rf $output
touch $output

rm -rf perf.*

# Check if mode is dataset
if [ "$mode" == "dataset" ]; then
    for c in "${code[@]}"
    do
        for p in "${phase[@]}"
        do
            for s_i in $(seq 0 $((slices_len-1)))
            do
                s=${slices[$s_i]}
                for k in "${K[@]}"
                do
                    options=$k-$s-slices$s_i-$c-float32-$p
                    nsys profile -o perf -f true pytest -vs test_ops.py::test_perf["$options"]
                    nsys stats --report cuda_gpu_kern_sum perf.nsys-rep -f csv -o 1
                    echo "$options" >> $output
                    if [ "$c" == "fasten" ]; then
                        grep segment_matmul_kernel 1_cuda_gpu_kern_sum.csv >> $output
                        if [ "$p" == "backward" ]; then
                            grep split_matmul_kernel 1_cuda_gpu_kern_sum.csv >> $output
                        fi
                    elif [ "$c" == "pyg" ]; then
                        grep cutlass 1_cuda_gpu_kern_sum.csv >> $output
                    fi
                    tail -n 2 $output
                    rm 1_cuda_gpu_kern_sum.csv
                    rm -rf perf.*
                done
            done
        done
    done
elif [ "$mode" == "random" ]; then
    for c in "${code[@]}"
    do
        for p in "${phase[@]}"
        do
            for k in "${K[@]}"
            do
                for s in $(seq 100 200 2000)
                do
                    options=1000000-$s-$k-$c-float32-$p
                    nsys profile -o perf -f true pytest -vs test_ops.py::test_perf_random["$options"]
                    nsys stats --report cuda_gpu_kern_sum perf.nsys-rep -f csv -o 1
                    echo "$options" >> $output
                    if [ "$c" == "fasten" ]; then
                        grep segment_matmul_kernel 1_cuda_gpu_kern_sum.csv >> $output
                        if [ "$p" == "backward" ]; then
                            grep split_matmul_kernel 1_cuda_gpu_kern_sum.csv >> $output
                        fi
                    elif [ "$c" == "pyg" ]; then
                        grep cutlass 1_cuda_gpu_kern_sum.csv >> $output
                    fi
                    tail -n 2 $output
                    rm 1_cuda_gpu_kern_sum.csv
                    rm -rf perf.*
                done
            done
        done
    done
else
    echo "Invalid mode"
fi

cd .. || exit
