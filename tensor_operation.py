import torch


def clone():
    a = torch.arange(20).reshape(4, 5)
    b = a.clone()
    print("memory same?")
    print(id(a) == id(b))
    print("context same?")
    print(a == b)
    print("context:")
    print("a:")
    print(a)
    print("a + a.clone():")
    print(a + b)


def multiply():
    a = torch.arange(4).reshape(2, 2)
    b = torch.arange(4).reshape(2, 2)
    print(a)
    print(b)
    print(a * b)


def tensor_with_scalar():
    mat = torch.arange(6).reshape(2, 3)  # 生成一个2 * 3的矩阵
    sca = torch.tensor(2)  # 生成一个标量2
    print(mat)
    print(sca)
    print(mat * sca)
    print(mat + sca)


def sum_reduce_dim():
    mat = torch.arange(20).reshape(4, 5)
    vec = torch.arange(4)
    print(mat)
    print(vec)
    # 默认求所有元素和
    print(mat.sum())
    print(vec.sum())
    # 指定轴降维求和
    # 以矩阵为例
    print(mat.sum(axis=0))  # 沿行（轴-0）求和，即对于第i列，将它的每一行元素求和，即求每列的和
    print(mat.sum(axis=1))  # 沿列（轴-1）求和，即对于第i行，将它的每一列元素求和，即求每行的和
    print(mat.sum(axis=[0, 1]))  # 沿所有轴降维，即求全体和


def mean():
    mat = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    print(mat)
    print(mat.mean())
    print(mat.sum() / mat.numel())
    # 指定轴
    print("mean of column:")
    print(mat.mean(axis=0))  # 求每列的平均值
    print(mat.sum(axis=0) / mat.shape[0])
    print("mean of row:")
    print(mat.mean(axis=1))  # 求每行的平均值
    print(mat.sum(axis=1) / mat.shape[1])


def keep_dims():
    mat = torch.arange(20).reshape(4, 5)
    print("mat:")
    print(mat)
    print("sum of column:")
    print(mat.sum(axis=0, keepdims=True))  # 依然求每列的和，但求完和后仍体现为一个二维张量
    print("sum of row:")
    print(mat.sum(axis=1, keepdims=True))
    # 结果仍是两个轴，可以与原张量运算
    tmp = mat.sum(axis=1, keepdims=True)  # tmp为每行求和结果
    print("tmp:")
    print(tmp)
    # 利用广播机制求原矩阵除以按行求和的结果
    print("broadcast for mat / tmp:")
    print(mat / tmp)
    # 沿行（轴-0）计算矩阵中元素的累计值
    print("cumulative value of mat by row:")
    print(mat.cumsum(axis=0))


def dot_product():
    a = torch.arange(4)
    b = torch.tensor([1, 2, 5, 9])
    print(a)
    print(b)
    print("dot product:")
    print(torch.dot(a, b))
    print(torch.sum(a * b))
    print((a * b).sum())


def matrix_mul_vector():
    mat = torch.arange(20).reshape(5, 4)  # 生成一个5 * 4的矩阵
    vec = torch.arange(4)  # 生成一个4维向量
    print(mat)
    print(mat.shape)
    print(vec)
    print(vec.shape)
    print("mat * vec:")
    res = torch.mv(mat, vec)  # 矩阵乘向量
    print(res)
    print(res.shape)


def matrix_mul_matrix():
    mat1 = torch.arange(20).reshape(5, 4)  # 生成一个5 * 4的矩阵
    mat2 = torch.arange(12).reshape(4, 3)  # 生成一个4 * 3的矩阵
    print(mat1)
    print(mat1.shape)
    print(mat2)
    print(mat2.shape)
    print("mat1 * mat2:")
    res = torch.mm(mat1, mat2)
    print(res)
    print(res.shape)


def L1():
    vec = torch.tensor([1, -2, 3, -4])
    print(vec)
    print("L1:")
    print(torch.abs(vec).sum())


def L2():
    vec = torch.tensor([1., -2, 3, -4])
    print(vec)
    print("L2:")
    print(torch.norm(vec))
    print(torch.sqrt(torch.pow(vec, 2).sum()))  # 自己套公式


def matrix_l2():
    mat = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    print(mat)
    print("mat's L2 norm")
    print(torch.norm(mat))
    print(torch.sqrt(torch.pow(mat, 2).sum()))  # 自己套公式


if __name__ == '__main__':
    # clone()
    # multiply()
    # tensor_with_scalar()
    # sum_reduce_dim()
    # mean()
    # keep_dims()
    # dot_product()
    # matrix_mul_vector()
    # matrix_mul_matrix()
    # L1()
    # L2()
    matrix_l2()