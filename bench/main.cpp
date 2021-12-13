#include <benchmark/benchmark.h>
#include <numpy/arrayobject.h>
#include <Python.h>

#include "benches/openmp.hpp"
#include "benches/sequential.hpp"


BENCHMARK(bench_openmp_loops)
    ->RangeMultiplier(2)
    ->Range(8, 16384);
BENCHMARK(bench_sequential)
    ->RangeMultiplier(2)
    ->Range(8, 16384);


int main(int argc, char **argv)
{
    benchmark::Initialize(&argc, argv);

    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    Py_Initialize();
    import_array1(1);

    benchmark::RunSpecifiedBenchmarks();

    Py_Finalize();
    benchmark::Shutdown();

    return 0;
}
