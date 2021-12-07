add_subdirectory(api)
add_subdirectory(impl)

set(SRCS
    dummy.c
)
set(LIBS
  PRIVATE
    ${PROJECT_NAME}-api
    ${PROJECT_NAME}-impl
)

add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} ${LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dotopt"
    LINKER_LANGUAGE C
)
