set(SRCS
    openmp.cpp
    sequential.cpp
)
set(LIBS
    benchmark::benchmark
    Python3::NumPy
    Python3::Python

    ${PROJECT_NAME}-impl
)

add_library(${PROJECT_NAME}-bench-benches STATIC ${SRCS})
target_link_libraries(${PROJECT_NAME}-bench-benches ${LIBS})
target_compile_definitions(${PROJECT_NAME}-bench-benches PRIVATE
    -DPY_ARRAY_UNIQUE_SYMBOL=dotopt_api
    -DNO_IMPORT_ARRAY
    -DNPY_NO_DEPRECATED_API
)
