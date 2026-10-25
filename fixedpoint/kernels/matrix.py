"""满秩小矩阵的整数乘积与主元消元，不调用浮点 linalg 求解。"""
import numpy as np

from ..array import FixedPointArray
from .. import arithmetic as ar


def _item(value, index):
    return FixedPointArray(np.asarray(value.real[index]), value.format,
                           None if value.imag is None else np.asarray(value.imag[index]))


def conjugate_transpose(value):
    return ar.from_scaled(value.real.T, None if value.imag is None else -value.imag.T,
                          value.format.fractional_length, value.format)


def fixed_matmul(a, b, target, event=None):
    if a.real.ndim != 2 or b.real.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("fixed_matmul requires compatible two-dimensional matrices")
    shape = (a.shape[0], b.shape[1])
    real = np.zeros(shape, dtype=np.int64)
    complex_result = a.imag is not None or b.imag is not None
    imag = np.zeros(shape, dtype=np.int64) if complex_result else None
    for row in range(shape[0]):
        for column in range(shape[1]):
            acc = FixedPointArray(np.asarray(0, dtype=np.int64), target,
                                  np.asarray(0, dtype=np.int64) if complex_result else None)
            for inner in range(a.shape[1]):
                product = ar.mul(_item(a, (row, inner)), _item(b, (inner, column)), target, event)
                acc = ar.add(acc, product, target, event)
            real[row, column] = acc.real
            if imag is not None:
                imag[row, column] = 0 if acc.imag is None else acc.imag
    return FixedPointArray(real, target, imag)


def fixed_solve(a, b, target, event=None):
    if a.real.ndim != 2 or a.shape[0] != a.shape[1] or b.real.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("fixed_solve requires square A and a compatible matrix B")
    a, b = ar.requantize(a, target, event), ar.requantize(b, target, event)
    n, columns = a.shape[0], b.shape[1]
    matrix = [[_item(a, (r, c)) for c in range(n)] for r in range(n)]
    rhs = [[_item(b, (r, c)) for c in range(columns)] for r in range(n)]
    minus_one = ar.quantize(-1., target)
    def subtract(left, right):
        return ar.add(left, ar.mul(right, minus_one, target, event), target, event)
    def magnitude_squared(value):
        return int(value.real) ** 2 + (int(value.imag) ** 2 if value.imag is not None else 0)
    for pivot_column in range(n):
        pivot_row = max(range(pivot_column, n), key=lambda r: magnitude_squared(matrix[r][pivot_column]))
        if magnitude_squared(matrix[pivot_row][pivot_column]) == 0:
            raise np.linalg.LinAlgError("fixed-point matrix is singular after quantization; increase precision or use boundary")
        matrix[pivot_column], matrix[pivot_row] = matrix[pivot_row], matrix[pivot_column]
        rhs[pivot_column], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_column]
        pivot = matrix[pivot_column][pivot_column]
        matrix[pivot_column] = [ar.div(v, pivot, target, event) for v in matrix[pivot_column]]
        rhs[pivot_column] = [ar.div(v, pivot, target, event) for v in rhs[pivot_column]]
        for row in range(n):
            if row == pivot_column:
                continue
            factor = matrix[row][pivot_column]
            matrix[row] = [subtract(v, ar.mul(factor, p, target, event))
                           for v, p in zip(matrix[row], matrix[pivot_column])]
            rhs[row] = [subtract(v, ar.mul(factor, p, target, event))
                        for v, p in zip(rhs[row], rhs[pivot_column])]
    real = np.array([[int(v.real) for v in row] for row in rhs], dtype=np.int64)
    imag = np.array([[int(v.imag) if v.imag is not None else 0 for v in row] for row in rhs], dtype=np.int64)
    return FixedPointArray(real, target, imag if a.imag is not None or b.imag is not None else None)


def fixed_pseudoinverse_full_rank(a, target, event=None):
    """满列秩/满行秩的 Moore-Penrose 公式；秩亏不隐式正则化。"""
    hermitian = conjugate_transpose(a)
    if a.shape[0] >= a.shape[1]:
        gram = fixed_matmul(hermitian, a, target, event)
        return fixed_solve(gram, hermitian, target, event)
    gram = fixed_matmul(a, hermitian, target, event)
    identity = ar.quantize(np.eye(a.shape[0]), target)
    inverse = fixed_solve(gram, identity, target, event)
    return fixed_matmul(hermitian, inverse, target, event)
