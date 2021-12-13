add_subdirectory(benches)

set(SRCS
    main.cpp
)
set(LIBS
  PRIVATE
    benchmark::benchmark
    Python3::NumPy
    Python3::Python

    ${PROJECT_NAME}-bench-benches
)

add_executable(${PROJECT_NAME}-bench ${SRCS})
target_link_libraries(${PROJECT_NAME}-bench ${LIBS})
target_compile_definitions(${PROJECT_NAME}-bench PRIVATE
    -DPY_ARRAY_UNIQUE_SYMBOL=dotopt_api
    -DNPY_NO_DEPRECATED_API
)

add_custom_target(bench
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-bench --benchmark_perf_counters=cycles,instructions,cache-misses --benchmark_counters_tabular=true
    DEPENDS ${PROJECT_NAME}-bench
    SOURCES ${SRCS}
)
