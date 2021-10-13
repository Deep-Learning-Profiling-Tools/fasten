# FASTEN: A *Fast* Library for H*e*teroge*n*ous Graphs

## Introduction

## Installation

### Required packages

- torch  
- torch-geometric
- pytest

### Optional packages

- spack
- MAGMA

### Build instructions

```bash
mkdir build && cd build
../bin/build.sh
```

## Examples

- RGCN

```bash
cd examples/rgcn
# Without fasten
python rgcn.py --dataset AIFB
# With fasten
python rgcn.py --dataset AIFB -fasten
```
