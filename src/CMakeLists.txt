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
target_compile_definitions(${PROJECT_NAME}-api PRIVATE
    -DPY_ARRAY_UNIQUE_SYMBOL=dotopt_api
    -DNO_IMPORT_ARRAY
    -DNPY_NO_DEPRECATED_API
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dotopt"
    LINKER_LANGUAGE C
)
