set(SRCS
    api.c
    hello.c
)
set(LIBS
  PRIVATE
    Python3::Module
)

add_library(${PROJECT_NAME}-api OBJECT ${SRCS})
target_link_libraries(${PROJECT_NAME}-api ${LIBS})
