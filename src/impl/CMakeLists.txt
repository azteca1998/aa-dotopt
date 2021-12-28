set(SRCS
    imts.c
    sequential_asm.c
    sequential_asm.s
    sequential_asm_zorder.s
    sequential.c
    util.c
)
set(LIBS
  PRIVATE
    OpenMP::OpenMP_C
)

add_library(${PROJECT_NAME}-impl OBJECT ${SRCS})
target_link_libraries(${PROJECT_NAME}-impl ${LIBS})
set_target_properties(${PROJECT_NAME}-impl PROPERTIES
    # Add `-fopt-info-vec-all` to display autovectorization info.
    COMPILE_FLAGS "-Ofast -mtune=native -march=native"
)
