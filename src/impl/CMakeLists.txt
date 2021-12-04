set(SRCS
    sequential.c
)
set(LIBS
  PRIVATE
    Python3::Module
)

add_library(${PROJECT_NAME}-impl OBJECT ${SRCS})
target_link_libraries(${PROJECT_NAME}-impl ${LIBS})
