add_subdirectory(api)

set(SRCS
    dummy.c
)
set(LIBS
  PRIVATE
    ${PROJECT_NAME}-api
)

add_library(${PROJECT_NAME} SHARED ${SRCS})
target_link_libraries(${PROJECT_NAME} ${LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dotopt"
    LINKER_LANGUAGE C
)
