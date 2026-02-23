# gpu-bitonic-sort-cuda
CUDA implementation of Bitonic Sort (a sorting network) that runs compare–swap stages on the GPU.  
Includes optional per stage printing to visualize how the algorithm transforms the array.

# Features
- GPU kernel for bitonic compare–swap stages
- Works for `N` as a power of 2 (2, 4, 8, 16, ...)
- Optional stage-by-stage output (`k`, `j`) for learning/debugging

## Requirements
- NVIDIA GPU + CUDA Toolkit installed
- `nvcc` available in your PATH
