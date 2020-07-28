import unittest
import torch
import torch.optim as optim

class TestOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_param_groups(self):
        print("\ntest_param_groups")
        w1 = torch.randn(2, 2)
        w1.requires_grad = True

        w2 = torch.randn(2, 2)
        w2.requires_grad = True

        w3 = torch.randn(2, 2)
        w3.requires_grad = True

        optimizer = optim.SGD(params=[w1, w2], lr=0.1, momentum=0.9, weight_decay=0.001)

        print('len(optimizer.param_groups): ', len(optimizer.param_groups))
        print(optimizer.param_groups, '\n')

        optimizer_2 = optim.SGD([{'params': w1, 'lr': 0.1},
                                 {'params': w2, 'lr': 0.001}])
        print('len(optimizer_2.param_groups): ', len(optimizer_2.param_groups))
        print(optimizer_2.param_groups, '\n')

    def test_zero_grad(self):
        print("\ntest_zero_grad")
        w1 = torch.randn(2, 2)
        w1.requires_grad = True

        w2 = torch.randn(2, 2)
        w2.requires_grad = True

        optimizer = optim.SGD(params=[w1, w2], lr=0.1, momentum=0.9, weight_decay=0.001)

        optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

        print('参数w1的梯度：')
        print(optimizer.param_groups[0]['params'][0].grad, '\n')  # 参数组，第一个参数(w1)的梯度

        optimizer.zero_grad()
        print('执行zero_grad()之后，参数w1的梯度：')
        print(optimizer.param_groups[0]['params'][0].grad)  # 参数组，第一个参数(w1)的梯度

    def test_state_dict(self):
        print("\ntest_state_dict")
        import torch.nn as nn
        import torch.nn.functional as F

        # ----------------------------------- state_dict
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 1, 3)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(1 * 3 * 3, 2)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = x.view(-1, 1 * 3 * 3)
                x = F.relu(self.fc1(x))
                return x

        net = Net()

        # 获取网络当前参数
        net_state_dict = net.state_dict()

        print('net_state_dict类型：', type(net_state_dict))
        print('net_state_dict管理的参数: ', net_state_dict.keys())
        for key, value in net_state_dict.items():
            print('参数名: ', key, '\t大小: ', value.shape)

    def test_add_param_groups(self):
        print("\ntest_add_param_groups")
        import torch.nn as nn
        w1 = torch.randn(2, 2)
        w1.requires_grad = True

        w2 = torch.randn(2, 2)
        w2.requires_grad = True

        w3 = torch.randn(2, 2)
        w3.requires_grad = True

        # 一个参数组
        optimizer_1 = optim.SGD([w1, w2], lr=0.1)
        print('当前参数组个数: ', len(optimizer_1.param_groups))
        print(optimizer_1.param_groups, '\n')

        # 增加一个参数组
        print('增加一组参数 w3\n')
        optimizer_1.add_param_group({'params': w3, 'lr': 0.001, 'momentum': 0.8})

        print('当前参数组个数: ', len(optimizer_1.param_groups))
        print(optimizer_1.param_groups, '\n')

        print('可以看到，参数组是一个list，一个元素是一个dict，每个dict中都有lr, momentum等参数，这些都是可单独管理，单独设定，十分灵活！')


if __name__ == "__main__":
    unittest.main()