import torch


def torch_tensor():
    x = torch.arange(12)
    print(x)
    print(x.shape)
    print(x.numel())
    print(x.reshape(3, 4))
    print(torch.zeros((2, 3, 4)))  # 1个张量tensor，3 * 4 的零矩阵有2个
    print(torch.ones((2, 3, 4)))  # 同上，不过填入1
    print(torch.randn(3, 4))  # 采样自标准正态分布的随机数，3 * 4的矩阵
    print(torch.tensor([[1, 2, 3, 4], [6, 5, 7, 8], [2, 3, 4, 1]]))  # 手动初始化一个矩阵
    print(torch.zeros_like(x))  # 复制x, 并填0


def tensor_operation():
    x = torch.tensor([1, 2, 3, 3.6])
    y = torch.tensor([0.5, 4, 2, 0.1])
    print(x * y)
    print(x / y)
    print(x + y)
    print(x - y)
    print(x ** y)
    print(torch.exp(x))
    print(x == y)
    print(x.sum())


def tensor_concat():
    x = (torch.arange(12, dtype=torch.float32)).reshape(3, 4)
    y = (torch.arange(12)).reshape(3, 4)
    # 按行(轴-0)连接
    print(torch.cat((x, y), dim=0))
    # 按列(轴-1)连接
    print(torch.cat((x, y), dim=1))


def tensor_broadcast():
    x = torch.arange(3).reshape(3, 1)
    y = torch.arange(2).reshape(1, 2)
    print(x)
    print(y)
    # add x and y
    print(x + y)


def index_slice():
    x = torch.arange(12).reshape(3, 4)
    print("initial:\n", x)
    print("last:\n", x[-1])  # last line
    print("row 1 ~ 2:\n", x[1:3])  # 第 1 ~ 2 行
    print("row 0 ~ 1 and column 1 ~ 2:\n", x[0:2, 1:3])  # 第 0 ~ 1 行，第 1 ~ 2 列
    print("column 2:\n", x[:, 2])  # 第 2 列
    print("row 1:\n", x[1, :])  # 第 1 行
    print("row 1 and col 2:\n", x[1, 2], x[1][2])  # 第 1 行，第 2 列
    x[1, 2] = 114  # 修改
    print("modify:\n", x)


def tensor_memory():
    x = torch.arange(4)
    y = torch.arange(4)
    old = id(y)
    y = x + y
    print(old == id(y))
    print("slice operation for reusing memory")
    # slice operation for reusing memory
    old = id(x)
    x[:] = x + y
    print(old == id(x))
    x += y
    print(old == id(x))


def tensor_transform():
    x = torch.arange(4)
    print(x)
    y = x.numpy()
    print(y)
    print(type(x), type(y))
    # 大小为 1 的张量可以转化为 python标量
    z = torch.tensor([1.14])
    print(z, z.item(), float(z), int(z))  # 转化成标量可以使用 item()，也可以使用python内置函数 float() 和 int()


if __name__ == "__main__":
    # torch_tensor()
    tensor_operation()
    # tensor_concat()
    # tensor_broadcast()
    # index_slice()
    # tensor_memory()
    tensor_transform()
