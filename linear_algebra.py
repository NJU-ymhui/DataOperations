import torch


def scalar():
    x = torch.tensor(3)  # 生成一个标量3
    y = torch.tensor(2.5)  # 生成一个标量2.5
    z = torch.tensor(1.)
    print(x, y, z)
    # calculate
    print(x + y + z, x * y / z, x ** y, x % z)

    # differ
    x = torch.tensor([3])  # 注：生成的不是标量，而是一个长度为 1 的一维张量
    print(x)


def vector():
    vec = torch.arange(4)  # 生成一个长度为 4 的一维张量表示向量，arange 函数生成一个从 0 到 3 的序列
    print(vec)
    # visit element in vec
    print(vec[0], vec[1], vec[2], vec[3])
    # change element in vec
    vec[0] = 1
    print(vec[0])
    print(vec)
    # show the length, dim of vec
    print("size:", vec.size(), "shape:", vec.shape)  # 因为向量vec是一维张量，所以它的shape就是这个向量的维数
    # use len()
    print("size by len():", len(vec))


def matrix():
    mat = torch.arange(20).reshape(5, 4)  # 生成一个普通矩阵5行4列，0-19
    print(mat)
    # 通过matrix的T属性访问其转置矩阵
    print("transpose of matrix:")
    print(mat.T)


def tensor():
    print("3 * 4 * 5:")
    tens = torch.arange(60).reshape(3, 4, 5)  # 生成一个三维张量，4行5列的矩阵有3个
    print(tens)
    print("2 x 3 x 4 x 5:")
    tens2 = torch.arange(120).reshape(2, 3, 4, 5)  # 生成一个四维张量，4行5列的矩阵有3个, 这样的组合还有两个
    print(tens2)


if __name__ == "__main__":
    # scalar()
    # vector()
    # matrix()
    tensor()