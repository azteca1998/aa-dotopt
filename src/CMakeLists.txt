add_subdirectory(api)
add_subdirectory(impl)

set(SRCS
    imts.c
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

add_library(${PROJECT_NAME}-pylib SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME}-pylib ${LIBS})
set_target_properties(${PROJECT_NAME}-pylib PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dotopt"
    LINKER_LANGUAGE C
)
