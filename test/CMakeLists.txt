function(add_python_test TEST_NAME TEST_MAIN)
    add_test(
        NAME ${TEST_NAME}
        COMMAND python3 -c "import sys; sys.path.append('${CMAKE_BINARY_DIR}/src'); exec(open('${TEST_MAIN}', 'rt').read())"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_python_test(hello hello.py)
