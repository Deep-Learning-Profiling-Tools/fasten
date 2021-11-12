# FASTEN: A *Fast* Library for Heterog*en*eous Graphs

## Introduction

## Installation

### Required packages

```bash
pip install -U torch torch-geometric pytest
```

### Optional packages

```bash
spack install magma
```

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
