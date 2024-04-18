# FASTEN: A Library of *Fast Segment* Operators

## Introduction
FASTEN is a library aimed at speeding up Heterogeneous Graph Neural Network (HGNN) workloads.   
The current version of FASTEN focuses on improving segmented matrix multiplication, a critical operator in HGNNs.
Fasten implements a simple interface, making it easy to integrate with existing graph library PyG with minimal changes.  

## Installation

### Build Instructions

Install pytorch nightly and triton nightly. We use relatively new triton features so old triton releases may crash.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

```bash
git clone https://github.com/Jokeren/fasten.git && cd fasten
pip install .
```

## Examples
FASTEN's segment matrix multiplication operator has been integrated with various HGNN architecture such as *RGCN*, *HGT*, *RGAT* in PyG.  
Examples on how to run the examples can be found below:

### GNN Examples

- RGCN

```bash
cd examples/rgcn
# Without fasten
# Available datasets are: AIFB, MUTAG, BGS, AM
python rgcn.py --device cuda --dataset AIFB
# With fasten
python rgat.py --device cuda --mode fasten --dataset AIFB
```

- HGT

```bash
cd examples/rgcn
# Without fasten
# Available datasets are: DBLP, Freebase, AIFB, MUTAG, BGS, AM
python rgcn.py --device cuda --dataset DBLP
# With fasten
python rgat.py --device cuda --mode fasten --dataset DBLP
```

- RGAT

```bash
cd examples/rgat
# Without fasten
# Available datasets are: AIFB, MUTAG, BGS, AM
python rgat.py --device cuda --dataset MUTAG
# With fasten
python rgat.py --device cuda --mode fasten --dataset MUTAG
```

### Benchmarking

```bash
cd test
pytest -vs test_op.py::test_perf
```

## Compatibility

### Supported Platforms

- Linux

### Supported Hardware

- NVIDIA GPUs (Compute Capability 7.0+)

### Software requirements
- Pytorch >=2.2.0
- Triton >=3.0.0
- PyG >=2.6.0

## Publication

```
Keren Zhou, Karthik Ganapathi Subramanian, Po-Hsun Lin, Matthias Fey, Binqian Yin, and Jiajia Li. 2024.
FASTEN: Fast GPU-accelerated Segmented Matrix Multiplication for Heterogeneous Graph Neural Networks.
In Proceedings of the 38th ACM International Conference on Supercomputing (ICS’24), June 4–7, 2024, Kyoto, Japan.
```
