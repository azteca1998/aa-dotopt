function(add_python_test TEST_NAME TEST_MAIN)
    add_test(
        NAME ${TEST_NAME}
        COMMAND python3 -c "import sys; sys.path.append('${CMAKE_BINARY_DIR}/src'); exec(open('${TEST_MAIN}', 'rt').read())"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_python_test(dot_imts_sequential             imts_sequential.py)
add_python_test(dot_imts_sequential_asm         imts_sequential_asm.py)
add_python_test(dot_imts_sequential_zorder      imts_sequential_zorder.py)
add_python_test(dot_imts_sequential_asm_zorder  imts_sequential_asm_zorder.py)
add_python_test(dot_sequential                  sequential.py)
add_python_test(dot_sequential_asm              sequential_asm.py)
add_python_test(dot_sequential_asm_zorder       sequential_asm_zorder.py)
add_python_test(dot_sequential_zorder           sequential_zorder.py)
