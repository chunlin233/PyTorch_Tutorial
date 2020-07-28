import os, sys
sys.path.append(os.getcwd())
import unittest
import torch
import torchvision.transforms as transforms
import glob
import shutil, random, re, numpy
from PIL import Image


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_dataEncodeDecode(self):
        print("\ntest_dataEncodeDecode")
        # Unicode字符串
        string = "你好"
        # Unicode字符
        snowman = '\u2603'
        # 编码：将字符串转化为一系列字节的过程
        # utf-8 是一种通用的动态编码，可以表达所有的Unicode
        ds = snowman.encode('utf-8')
        # ascii 编码，将ascii字符编码为一个字节
        ds1 = "A".encode('ascii')
        ds2 = "A".encode('utf-8')
        print(string, type(string), len(string))
        print(snowman, type(snowman), len(snowman))
        print(ds, type(ds), len(ds))
        print(ds1, type(ds1), len(ds1))
        print(ds2, type(ds2), len(ds2))

        # 解码：将字节序列转化为Unicode字符串的过程
        # 我们从外界文本源（文本、数据库、网站等）获取的所有文本都是经过编码的字节串
        # 因此解码前需要知道它是以何种方式编码的
        place = 'caf\u00e9'
        print("\n%20s, %20s, %5d" % (place, type(place), len(place)))
        place_bytes = place.encode('utf-8')
        print("%20s, %20s, %5d" % (place_bytes, type(place_bytes), len(place_bytes)))
        place1 = place_bytes.decode('utf-8')
        print("%20s, %20s, %5d" % (place1, type(place1), len(place1)))
        place1 = place_bytes.decode('latin-1')
        print("%20s, %20s, %5d" % (place1, type(place1), len(place1)))
        place1 = place_bytes.decode('windows-1252')
        print("%20s, %20s, %5d" % (place1, type(place1), len(place1)))
        # place1 = place_bytes.decode('ascii')

    def test_torchMax(self):
        print("\ntest_torchMax")
        x = torch.randint(0, 10, (3, 5))
        print(x)
        values, indices = torch.max(x, dim=1)
        print(values)
        print(indices)

    def test_walk(self):
        print("\ntest_walk")
        train_dir = os.path.join('..', 'Data', 'train')
        for root, dirs, files in os.walk(train_dir, topdown=True):
            print(root, dirs)

    def test_split(self):
        print("\ntest_split")
        dataset_dir = os.path.join('..', 'Data', 'cifar-10-png', 'raw_test')
        train_dir = os.path.join("..", "..", "Data", "train")
        out_dir = os.path.join(train_dir, '0')
        # for root, dirs, files in os.walk(dataset_dir):
        #     print(root, dirs)
        imgs_list = glob.glob(os.path.join(dataset_dir, '0', '*.png'))
        print(imgs_list)
        print(os.path.split(dataset_dir))
        print(os.path.split(imgs_list[0]))
        print(os.path.join(out_dir, os.path.split(imgs_list[0])[-1]))

        print(os.path.splitext(imgs_list[0]))

        random.seed(666)
        random.shuffle(imgs_list)
        # shutil.copy(imgs_list[0], '.')

    def test_readtext(self):
        print("\ntest_readtext")
        text_path = os.path.join('..', 'Data', 'train.txt')
        imgpaths_labels = []
        with open(text_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgpaths_labels.append((words[0], int(words[1])))
        print(imgpaths_labels[:10])
        index = 1
        img_path, label = imgpaths_labels[index]
        print("\\", '\\\\')
        print(img_path)
        m = re.match(r'..\\(.*)', img_path)
        print(m.groups()[0])

        img_path = m.groups()[0]
        img = Image.open(img_path)

        print("img before convert: ", numpy.array(img).shape)
        img1 = Image.open(img_path).convert('RGB')
        print("img1 after convert: ", numpy.array(img1).shape)

        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        transform = transforms.ToTensor()
        print("img after transform ToTensor: ", transform(img1).shape)

    def test_variable(self):
        print("\ntest_Variable")
        from torch.autograd import Variable
        x = torch.randn(10, 3, 4, 4)
        print(type(x), x.requires_grad)
        # Variable可以让张量运算在GPU中运行
        x_v = Variable(x)
        print(type(x_v), x_v.requires_grad)

    def test_format(self):
        print("\ntest_format")
        string1 = 'Iteration[{:0>3}/{:0>3}]'
        print(string1.format(3, 10000))

    def test_time(self):
        print("\ntest_time")
        from datetime import datetime
        now_time = datetime.now()
        print(now_time)
        time_str = datetime.strftime(now_time, "%m-%d_%H-%M-%S")
        print(time_str)

    def test_getpng(self):
        print("\ntest_getpng")
        # 获取类别文件夹下所有png图片的路径
        data_dir = os.path.join('..', 'Data', 'cifar-10-png', 'raw_test')
        # method 1
        for root, dirs, files in os.walk(data_dir):
            for sDir in dirs:
                imgs_list = glob.glob(os.path.join(root, sDir, '*.png'))
        # method 2
        for root, dirs, files in os.walk(data_dir):
            for sDir in dirs:
                i_dir = os.path.join(root, sDir)        # 获取各类的文件夹 相对路径
                img_list = os.listdir(i_dir)            # 获取类别文件夹下所有文件的路径
                for i in range(len(img_list)):
                    if not img_list[i].endswith('png'): # 若不是png文件，跳过
                        continue



if __name__ == "__main__":
    unittest.main()