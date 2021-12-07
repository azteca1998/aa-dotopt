import dotopt
import numpy as np


def run_subtest(name, m, k, n):
    print(f'Subtest {name}: M, K, N = {m}, {k}, {n}')
    op_a = np.random.random((m, k)).astype(np.float32)
    op_b = np.random.random((k, n)).astype(np.float32)

    res_obtained = dotopt.dot_sequential(op_a, op_b)
    res_expected = np.dot(op_a, op_b)
    assert np.allclose(res_obtained, res_expected)
    print('  Success!')


run_subtest('A', 8, 8, 8)
run_subtest('B', 8, 8, 12)
run_subtest('C', 100, 200, 100)
