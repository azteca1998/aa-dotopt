set(SRCS
    sequential.cpp
    sequential_asm.cpp
)
set(LIBS
    benchmark::benchmark

    ${PROJECT_NAME}-impl
)

add_library(${PROJECT_NAME}-bench-benches STATIC ${SRCS})
target_link_libraries(${PROJECT_NAME}-bench-benches ${LIBS})
