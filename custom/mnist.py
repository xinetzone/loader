'''
作者：xinetzone
时间：2019/12/3
'''
import struct
from pathlib import Path
import numpy as np
import gzip


class MNIST:
    def __init__(self, root, namespace):
        """MNIST 与 FASGION-MNIST 数据解码工具
        1. (MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist) 下载后放置在 `mnist` 目录
        2. (A dataset of Zalando's article images consisting of fashion products,
        a drop-in replacement of the original MNIST dataset from https://github.com/zalandoresearch/fashion-mnist) 
            数据下载放置在 `fashion-mnist` 目录

        Each sample is an image (in 2D NDArray) with shape (28, 28).

        

        参数
        ========
        root : 数据根目录，如 'E:/Data/Zip/'
        namespace : 'mnist' or 'fashion-mnist'

        实例属性
        ========
        train_data：训练数据集图片
        train_label：训练数据集标签名称
        test_data：测试数据集图片
        test_label：测试数据集标签名称
        """
        root = Path(root) / namespace
        self._name2array(root)

    def _name2array(self, root):
        '''
        官方网站是以 `[offset][type][value][description]` 的格式封装数据的，
            因而我们使用 `struct.unpack`
        '''
        _train_data = root / 'train-images-idx3-ubyte.gz'  # 训练数据集文件名
        _train_label = root / 'train-labels-idx1-ubyte.gz'  # 训练数据集的标签文件名
        _test_data = root / 't10k-images-idx3-ubyte.gz'   # 测试数据集文件名
        _test_label = root / 't10k-labels-idx1-ubyte.gz'  # 测试数据集的标签文件名
        self.train_data = self.get_data(_train_data)  # 获得训练数据集图片
        self.train_label = self.get_label(_train_label)  # 获得训练数据集标签名称
        self.test_data = self.get_data(_test_data)   # 获得测试数据集图片
        self.test_label = self.get_label(_test_label)  # 获得测试数据集标签名称

    def get_data(self, data):
        '''获取图像信息'''
        with gzip.open(data, 'rb') as fin:
            shape = struct.unpack(">IIII", fin.read(16))[1:]
            data = np.frombuffer(fin.read(), dtype=np.uint8)
        data = data.reshape(shape)
        return data

    def get_label(self, label):
        '''获取标签信息'''
        with gzip.open(label, 'rb') as fin:
            struct.unpack(">II", fin.read(8))  # 参考数据集的网站，即 offset=8
            # 获得数据集的标签
            label = fin.read()
        label = np.frombuffer(label, dtype=np.uint8).astype(np.int32)
        return label