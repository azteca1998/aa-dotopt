set(SRCS
    api.c
    hello.c
    sequential.c
)
set(LIBS
  PRIVATE
    Python3::Module
    Python3::NumPy
)

add_library(${PROJECT_NAME}-api OBJECT ${SRCS})
target_link_libraries(${PROJECT_NAME}-api ${LIBS})
target_compile_definitions(${PROJECT_NAME}-api PRIVATE
    -DPY_ARRAY_UNIQUE_SYMBOL=dotopt_api
    -DNO_IMPORT_ARRAY
    -DNPY_NO_DEPRECATED_API
)
