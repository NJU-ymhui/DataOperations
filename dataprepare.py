import os
import pandas as pd
import torch


def create_data_read():
    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_file = os.path.join('.', 'data', 'house_tiny.csv')
    # create data
    with open(data_file, 'w') as f:
        #        房屋数量  巷子类型 房屋价格
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')  # NA为缺失值
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    # use pandas read data
    df = pd.read_csv(data_file)
    print(df)
    return df


def insert_missing(data):
    # data来自create_data_read
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # inputs 取前两列，outputs取第三列 Price(暂时用不到)
    print(inputs)

    # 为连续值的缺失值插值
    inputs = inputs.fillna(inputs.mean())  # 以均值替换NaN
    print("after inserting for continuous variable:")
    print(inputs)

    # 为离散值的缺失值插值
    inputs = pd.get_dummies(inputs, dummy_na=True)  # dummy_na=True, 表示为缺失值创建一个新特征
    print("after inserting for discrete variable:")
    print(inputs)
    return inputs, outputs


def transfer_tensor(data):
    # data来自insert_missing
    inputs, outputs = data
    x = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(outputs.to_numpy(dtype=float))
    print(x)
    print(y)


if __name__ == '__main__':
    transfer_tensor(insert_missing(create_data_read()))
