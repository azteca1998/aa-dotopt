# Benchmark Results

> NOTE: To obtain the real `Time` and `CPU` from multithreaded benchmarks,
>   divide each value by 16.

| Benchmark                   | Time      | CPU       | Iterations | Cache Misses | Cycles   | Instructions |
|-----------------------------|-----------|-----------|------------|--------------|----------|--------------|
| bench_imts_sequential/8     |   0.001ms |   0.001ms |    1218694 |       0.010  | 1881.86  |        4.98k |
| bench_imts_sequential/16    |   0.004ms |   0.002ms |     289022 |       0.037  | 7730.12  |       19.87k |
| bench_imts_sequential/32    |   0.021ms |   0.014ms |      49298 |       0.635  |   46.30k |      100.07k |
| bench_imts_sequential/64    |   0.191ms |   0.114ms |       6314 |       8.558  |  369.75k |      793.23k |
| bench_imts_sequential/128   |   1.800ms |   0.906ms |        757 |     287.724  |    2.93M |        6.33M |
| bench_imts_sequential/256   |  12.400ms |   7.250ms |         96 |      42.289k |   23.47M |       50.65M |
| bench_imts_sequential/512   |  92.900ms |  59.100ms |         12 |     412.875k |  191.33M |      404.30M |
| bench_imts_sequential/1024  | 644.000ms | 492.000ms |          2 |       8.079M |    1.59G |        3.23G |
| bench_imts_sequential/2048  |   7.595s  |   4.098s  |          1 |     188.887M |   13.23G |       25.86G |
| bench_imts_sequential/4096  |  73.941s  |  39.797s  |          1 |      11.666G |  128.61G |      206.90G |
| bench_imts_sequential/8192  |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |
| bench_imts_sequential/16384 |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |

| Benchmark                       | Time      | CPU       | Iterations | Cache Misses | Cycles   | Instructions |
|---------------------------------|-----------|-----------|------------|--------------|----------|--------------|
| bench_imts_sequential_asm/8     |   0.000ms |   0.000ms |    3138033 |       0.006  |  738.88  |        1.16k |
| bench_imts_sequential_asm/16    |   0.001ms |   0.001ms |    1020203 |       0.033  |    2.25k |        4.07k |
| bench_imts_sequential_asm/32    |   0.008ms |   0.005ms |     152188 |       1.202  |   14.74k |       27.40k |
| bench_imts_sequential_asm/64    |   0.072ms |   0.036ms |      19433 |      13.698  |  116.41k |      216.26k |
| bench_imts_sequential_asm/128   |   0.588ms |   0.292ms |       2427 |     232.597  |  944.32k |        1.73M |
| bench_imts_sequential_asm/256   |   3.930ms |   2.430ms |        288 |      15.138k |    7.84M |       13.87M |
| bench_imts_sequential_asm/512   |  32.300ms |  21.500ms |         33 |     314.168k |   69.20M |      111.07M |
| bench_imts_sequential_asm/1024  | 493.000ms | 223.000ms |          3 |       2.692M |  718.11M |      888.91M |
| bench_imts_sequential_asm/2048  |   3.315s  |   1.889s  |          1 |      73.866M |    6.09G |        7.11G |
| bench_imts_sequential_asm/4096  |  27.034s  |  16.432s  |          1 |       1.435G |   53.04G |       56.91G |
| bench_imts_sequential_asm/8192  |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |
| bench_imts_sequential_asm/16384 |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |

| Benchmark                                 | Time      | CPU       | Iterations | Cache Misses | Cycles   | Instructions |
|-------------------------------------------|-----------|-----------|------------|--------------|----------|--------------|
| bench_imts_sequential_asm_omp_tasks/8     | 122.000ms |  10.600ms |         49 |     721.327  |   32.88M |       30.98M |
| bench_imts_sequential_asm_omp_tasks/16    |  91.800ms |   9.960ms |         55 |     807.636  |   31.54M |       31.60M |
| bench_imts_sequential_asm_omp_tasks/32    | 111.000ms |  14.800ms |         55 |       3.488k |   42.15M |       40.64M |
| bench_imts_sequential_asm_omp_tasks/64    | 651.000ms |  74.200ms |         10 |      29.316k |  229.19M |      197.22M |
| bench_imts_sequential_asm_omp_tasks/128   | 424.000ms |  58.600ms |         26 |     181.558k |  194.49M |      169.90M |
| bench_imts_sequential_asm_omp_tasks/256   |   6.969s  | 929.000ms |          1 |       1.525M |    3.22G |        2.81G |
| bench_imts_sequential_asm_omp_tasks/512   |  33.603s  |   5.860s  |          1 |      16.384M |   18.39G |       14.29G |

| bench_imts_sequential_asm_omp_tasks/1024  | 644.000ms | 492.000ms |          2 |       8.079M |    1.59G |        3.23G |
| bench_imts_sequential_asm_omp_tasks/2048  |   7.595s  |   4.098s  |          1 |     188.887M |   13.23G |       25.86G |
| bench_imts_sequential_asm_omp_tasks/4096  |  73.941s  |  39.797s  |          1 |      11.666G |  128.61G |      206.90G |
| bench_imts_sequential_asm_omp_tasks/8192  |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |
| bench_imts_sequential_asm_omp_tasks/16384 |     N/A   |     N/A   |        N/A |         N/A  |     N/A  |         N/A  |
