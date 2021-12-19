set(SRCS
    asm.c
    asm.s
    imts.c
    openmp.c
    sequential.c
    util.c
)
set(LIBS
  PRIVATE
    OpenMP::OpenMP_C
    Python3::Module
    Python3::NumPy
)

add_library(${PROJECT_NAME}-impl OBJECT ${SRCS})
target_link_libraries(${PROJECT_NAME}-impl ${LIBS})
target_compile_definitions(${PROJECT_NAME}-impl PRIVATE
    -DPY_ARRAY_UNIQUE_SYMBOL=dotopt_api
    -DNO_IMPORT_ARRAY
    -DNPY_NO_DEPRECATED_API
)
set_target_properties(${PROJECT_NAME}-impl PROPERTIES
    # Add `-fopt-info-vec-all` to display autovectorization info.
    COMPILE_FLAGS "-Ofast -mtune=native -march=native"
)
