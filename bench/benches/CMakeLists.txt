set(SRCS
    imts.cpp
    sequential.cpp
    sequential_asm.cpp
)
set(LIBS
    benchmark::benchmark
    OpenMP::OpenMP_CXX

    ${PROJECT_NAME}
)

add_library(${PROJECT_NAME}-bench-benches STATIC ${SRCS})
target_link_libraries(${PROJECT_NAME}-bench-benches ${LIBS})
