add_subdirectory(api)
add_subdirectory(impl)

set(SRCS
    imts.c
    zorder.c
)
set(LIBS
  PRIVATE
    Python3::Module
    Python3::NumPy
    Threads::Threads

    ${PROJECT_NAME}-api
    ${PROJECT_NAME}-impl
)

add_library(${PROJECT_NAME} STATIC ${SRCS})
target_link_libraries(${PROJECT_NAME} ${LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    # Add `-fopt-info-vec-all` to display autovectorization info.
    COMPILE_FLAGS "-Ofast -mtune=native -march=native"
)

add_library(${PROJECT_NAME}-pylib SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME}-pylib ${LIBS})
set_target_properties(${PROJECT_NAME}-pylib PROPERTIES
    # Add `-fopt-info-vec-all` to display autovectorization info.
    COMPILE_FLAGS "-Ofast -mtune=native -march=native"

    PREFIX ""
    OUTPUT_NAME "dotopt"
    LINKER_LANGUAGE C
)
