'''
作者：xinetzone
时间：2019/12/3
'''
import tarfile
from pathlib import Path
import pickle
import time
import numpy as np


class Cifar:
    def __init__(self, root,  namespace):
        """CIFAR image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html
        Each sample is an image (in 3D NDArray) with shape (3, 32, 32).

        参数
        =========
        meta : 保存了类别信息
        root : str, 数据根目录
        namespace : 'cifar-10' 或 'cifar-100'
        """
        #super().__init__(*args, **kwds)
        #self.__dict__ = self
        self.root = Path(root)
        # 解压数据集到 root，并将解析后的数据载入内存
        self._load(namespace)

    def _extractall(self, namespace):
        '''解压 tar 文件并返回路径

        参数
        ========
        tar_name：tar 文件名称
        '''
        tar_name = self.root / f'{namespace}-python.tar.gz'
        with tarfile.open(tar_name) as tar:
            tar.extractall(self.root)     # 解压全部文件
            names = tar.getnames()  # 获取解压后的文件所在目录
        return names

    def _decode(self, path):
        '''载入二进制流到内存'''
        with open(path, 'rb') as fp:  # 打开文件
            # 载入数据到内存
            data = pickle.load(fp, encoding='bytes')
        return data

    def _load_cifar10(self, names):
        '''将解析后的 cifar10 数据载入内存'''
        # 获取数据根目录
        R = [self.root /
             name for name in names if (self.root / name).is_dir()][0]
        # 元数据信息
        meta = self._decode(list(R.glob('*.meta'))[0])
        # 训练集信息
        train = [self._decode(path) for path in R.glob('*_batch_*')]
        # 测试集信息
        test = [self._decode(path) for path in R.glob('*test*')][0]
        return meta, train, test

    def _load_cifar100(self, names):
        '''将解析后的 cifar100 数据载入内存'''
        # 获取数据根目录
        R = [self.root /
             name for name in names if (self.root / name).is_dir()][0]
        # 元数据信息
        meta = self._decode(list(R.glob('*meta*'))[0])
        # 训练集信息
        train = [self._decode(path) for path in R.glob('*train*')][0]
        # 测试集信息
        test = [self._decode(path) for path in R.glob('*test*')][0]
        return meta, train, test

    def _load(self, namespace):
        # 解压数据集到 root，并返回文件列表
        names = self._extractall(namespace)
        if namespace == 'cifar-10':
            self.meta, train, test = self._load_cifar10(names)
            self.trainX = np.concatenate(
                [x[b'data'] for x in train]).reshape(-1, 3, 32, 32)
            self.trainY = np.concatenate([x[b'labels'] for x in train])
            self.testX = np.array(test[b'data']).reshape(-1, 3, 32, 32)
            self.testY = np.array(test[b'labels'])
        elif namespace == 'cifar-100':
            self.meta, train, test = self._load_cifar100(names)
            self.trainX = np.array(train[b'data']).reshape(-1, 3, 32, 32)
            self.testX = np.array(test[b'data']).reshape(-1, 3, 32, 32)
            self.train_fine_labels = np.array(train[b'fine_labels'])
            self.train_coarse_labels = np.array(train[b'coarse_labels'])
            self.test_fine_labels = np.array(test[b'fine_labels'])
            self.test_coarse_labels = np.array(test[b'coarse_labels'])