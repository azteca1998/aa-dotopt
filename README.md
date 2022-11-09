# DotOpt - Optimized matrix multiplication

## Development environment configuration

  1. Update the container.
    
    ./scripts/update-all.sh

  2. Install the required dependencies.
  
    ./scripts/install-deps.sh

  3. Fetch third party libraries
  
    ./scripts/fetch-libs.sh

> **WARNING**: Benchmarking requires `SYS_CAP_ADMIN`.

## Project overview

**Main library**

The main library is implemented as a python module which computes matrix 
multiplication operations using NumPy types.

**Testing**

Tests are implemented in Python, and added to `test/CMakeLists.txt` so that they
are executed when testing.

**Benchmarking**

Benchmarks are implemented in C/C++ using
[Google's benchmarking library](https://github.com/google/benchmark).

> Warning: CPU frequency scaling should be disabled during the benchmarks.

```sh
# To query the current CPU frequency scaler:
sudo cpupower frequency-info

# To force the CPU to run at full power:
sudo cpupower frequency-set --governor performance

# To restore the previous CPU fequency scaler profile:
sudo cpupower frequency-set --governor <previous governor>
```

## Algorithm implementations

  1. Sequential algorithm.

**OpenMP parallelism**

  2. Parallel algorithm using OpenMP's loop parallelism.
  3. Parallel algorithm using OpenMP's task parallelism.

  4. Parallel algorithm with row/col tiling using OpenMP's loop parallelism.
  5. Parallel algorithm with row/col tiling using OpenMP's task parallelism.

  6. Parallel algorithm with 2d tiling using OpenMP's loop parallelism.
  7. Parallel algorithm with 2d tiling using OpenMP's task parallelism.

**Our IMTS implementation**

  8. Parallel algorithm using 1 level IMTS with random tile ordering.
  9. Parallel algorithm using 1 level IMTS with tile ordering.

  10. Parallel algorithm using 2 level ITMS with random tile ordering.
  11. Parallel algorithm using 2 level ITMS with tile ordering.

  12. Parallel algorithm using 3 level ITMS with random tile ordering.
  13. Parallel algorithm using 3 level ITMS with tile ordering.

**Extras**

  14. The best version to date with a hand-crafted matrix multiplication
    assembly code.
  15. The hand-crafted version with 4-level ITMS tiling, where the smallest
    tiles fit into the vector registers. In other words, a hand crafted tiling
    level for our hand-crafted matrix multiplication implementation.

**GPU using Vulkan compute shaders**

  16. (Not implemented) A GPU version where each thread computes an output.
  17. (Not implemented) A tiling GPU version where each thread computes a 4x4 tile (GLSL).
  18. (Not implemented) A tiling GPU version where each SM computes a tile using the shared memory
    as a scratchpad for inputs and/or outputs.

**GPU using OpenCL (if we can make it work)**

  19. (Not implemented) Same as #16.
  20. (Not implemented) Same as #17.
  21. (Not implemented) Same as #18.

**External implementations (for reference when comparing)**

  22. (Testing only) NumPy implementation.
  23. (Not used) TensorFlow implementation.
  24. (Not used) OpenBLAS (maybe?).
  25. (Not used) clBLAS (maybe?).
  25. (Not used) ArrayFire (OpenCL; maybe?).

### Intelligent Multilevel Tiling Scheduler

The IMTS scheduler divides the workload into 2d tiles taking into account the
sizes of the LLC. Those are then passed on to another tiler which further
divides the workload into smaller tiles which fit into the next-to-last level
cache. The same process is repeated until the first level cache is reached.

The order in which the tiles are computed is predefined to minify the number of
transfers between the caches. For example, a cache will need both input tiles
and the output. When switching tiles, only one of those will be loaded, and
preferably one of the inputs, since the outputs will have to be transferred back
to the next level.
