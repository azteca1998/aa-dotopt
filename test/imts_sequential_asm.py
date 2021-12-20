import dotopt
import numpy as np


def run_subtest(name, m, k, n):
    print(f'Subtest {name}: M, K, N = {m}, {k}, {n}')
    op_a = np.random.random((m, k)).astype(np.float32)
    op_b = np.random.random((k, n)).astype(np.float32)

    res_obtained = dotopt.dot_imts_sequential_asm(op_a, op_b)
    res_expected = np.dot(op_a, op_b)
    assert np.allclose(res_obtained, res_expected)
    print('  Success!')


np.set_printoptions(precision=2, linewidth=120, formatter={'float_kind': '{:0.2f}'.format})

run_subtest('(8,8,8)', 8, 8, 8)

run_subtest('(16,8,8)', 16, 8, 8)
run_subtest('(8,16,8)', 8, 16, 8)
run_subtest('(16,16,8)', 16, 16, 8)
run_subtest('(8,8,16)', 8, 8, 16)
run_subtest('(16,8,16)', 16, 8, 16)
run_subtest('(8,16,16)', 8, 16, 16)
run_subtest('(16,16,16)', 16, 16, 16)

run_subtest('(12,8,8)', 12, 8, 8)
run_subtest('(8,12,8)', 8, 12, 8)
run_subtest('(12,12,8)', 12, 12, 8)
run_subtest('(8,8,12)', 8, 8, 12)
run_subtest('(12,8,12)', 12, 8, 12)
run_subtest('(8,12,12)', 8, 12, 12)
run_subtest('(8,12,12)', 12, 12, 12)

run_subtest('(100,200,100)', 100, 200, 100)
