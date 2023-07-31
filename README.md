# FASTEN: A Library of *Fast* S*e*gme*n*t Operators

## Introduction

## Installation

### Optional packages

```bash
spack install magma
```

### Build instructions

```bash
git clone https://github.com/Jokeren/fasten.git && cd fasten
pip install .
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
