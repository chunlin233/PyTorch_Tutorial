import os, sys
sys.path.append(os.getcwd())
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import glob
import shutil, random, re, numpy
from PIL import Image
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_state_dict(self):
        print("\ntest_state_dict")
        net = Net()
        net.initialize_weights()
        # 预训练好的网络参数dict
        pretrain_dict = torch.load('net_params.pkl')
        # 现在的网络参数dict
        net_state_dict = net.state_dict()
        # 剔除不匹配的权值参数
        pretrain_dict_1 = {k: v for k, v in pretrain_dict.items() if k in net_state_dict}
        # 更新新模型参数字典
        net_state_dict.update(pretrain_dict_1)
        # 将包含预训练模型参数的字典"放"到新模型中
        net.load_state_dict(net_state_dict)

        print(type(net))
        # for name, layer in net_state_dict.items():
        #     print(name, layer.shape)
        # torch.save(net.state_dict(), 'net_params.pkl')

    def test_output(self):
        print("\ntest_output")
        net = Net()
        data = torch.randn(2, 3, 32, 32)
        label = torch.randint(0, 9, (2,))
        print(label)
        input, label = Variable(data), Variable(label)
        print("input:", input.requires_grad)
        out = net(input)
        print("out:", out.requires_grad)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)
        print("loss:", loss)
        print("out.data:", out.data)
        out_1 = torch.max(out.data, dim=1)
        print("out_1:\n", out_1)
        out_2 = torch.max(out, dim=1)
        print("out_2:\n", out_2)

    def test_optimizer(self):
        print("\ntest_optimizer")
        net = Net()
        # 将fc3层的参数从原始网络参数中剔除
        ignored_params = list(map(id, net.fc3.parameters()))
        # 这里的map和filter的功能类似
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # 为fc3层设置需要的学习率
        optimizer = optim.SGD([
            {'params': base_params},
            {'params': net.fc3.parameters(), 'lr': 0.01 * 10}], 0.01, momentum=0.9, weight_decay=1e-4)
        print(len(optimizer.param_groups))
        for group in optimizer.param_groups:
            print(len(group.items()))
            for k, v in group.items():
                print(k, ":")
                if k in "params":
                    print([v1.shape for v1 in v])
                else:
                    print(v)


        # 返回了一个generator
        y = filter(lambda x: x>0, [-2, 3, 4])
        for i in y:
            print(i)

if __name__ == "__main__":
    unittest.main()