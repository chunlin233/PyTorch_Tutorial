import unittest
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from test_Model import Net

class TestLR(unittest.TestCase):
    def setUp(self):
        self.net = Net()
        self.ignored_params = list(map(id, self.net.fc3.parameters()))
        self.base_params = filter(lambda p: id(p) not in self.ignored_params, self.net.parameters())
        self.optimizer = optim.SGD([
            {'params': self.base_params},
            {'params': self.net.fc3.parameters(), 'lr': 0.001*100}
        ], lr=0.001, momentum=0.9, weight_decay=1e-4)


    def tearDown(self):
        pass

    def test_LambdaLR(self):
        print("\n------------------------ test_LambdaLR")
        lambda1 = lambda epoch: epoch // 3
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1, lambda2])
        for epoch in range(10):
            # lr = base_lr * lamda(self.last_epoch)
            scheduler.step()
            print('epoch:', epoch, 'lr:', scheduler.get_lr())

    def test_StepLR(self):
        print("\n------------------------ test_StepLR")
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        for epoch in range(30):
            scheduler.step()
            print('epoch: ', epoch, 'lr: ', scheduler.get_lr())

    def test_MultiStepLR(self):
        print("\n------------------------ MultiStepLR")
        scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15], gamma=0.1)
        for epoch in range(30):
            scheduler.step()
            print('epoch: ', epoch, 'lr: ', scheduler.get_lr())

if __name__ == "__main__":
    unittest.main()