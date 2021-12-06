import dotopt
import numpy as np


# TODO: Test (use `dotopt.dot_sequential(a, b)`).

print("Test 1: Matriz 8x8 * 8x8")
a= np.random.random((8,8)).astype(np.float)
b= np.random.random((8,8)).astype(np.float)

res_nump = np.dot(a,b)

res_seq=dotopt.dot_sequential(a,b)

print(np.allclose(res_nump,res_seq))

print("Test 2: Matriz 8x8 * 8x12")
a= np.random.random((8,8)).astype(np.float)
b= np.random.random((8,12)).astype(np.float)

res_nump = np.dot(a,b)

res_seq=dotopt.dot_sequential(a,b)

print(np.allclose(res_nump,res_seq))

print("Test 3: Matriz 100x200 * 200 * 100")
a= np.random.random((100,200)).astype(np.float)
b= np.random.random((200,100)).astype(np.float)

res_nump = np.dot(a,b)

res_seq=dotopt.dot_sequential(a,b)

print(np.allclose(res_nump,res_seq))


