# Optimization:

## Idea(s):
- The data is divided into blocks, which are processed by the GPU. To minimize memory misses, we conducted multiple tests to determine the most efficient **BLOCK_SIZE**. These tests help identify the optimal configuration for improved performance.
