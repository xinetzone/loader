'''
作者：xinetzone
时间：2019/12/3
'''
import tables as tb
import numpy as np

from custom.cifar import Cifar
from custom.mnist import MNIST


class Bunch(dict):
    def __init__(self, root, *args, **kwargs):
        """将数据  MNIST，Fashion MNIST，Cifar 10，Cifar 100 打包
        为 HDF5
        
        参数
        =========
        root : 数据的根目录
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.mnist = MNIST(root, 'mnist')
        self.fashion_mnist = MNIST(root, 'fashion-mnist')
        self.cifar10 = Cifar(root, 'cifar-10')
        self.cifar100 = Cifar(root, 'cifar-100')

    def _change(self, img):
        '''将数据由 (num, channel, h, w) 转换为 (num, h, w, channel)'''
        return np.transpose(img, (0, 2, 3, 1))

    def toHDF5(self, save_path):
        '''将数据打包为 HDF5 格式
        
        参数
        ===========
        save_path：数据保存的路径
        '''
        filters = tb.Filters(complevel=7, shuffle=False)
        with tb.open_file(f'{save_path}/X.h5', 'w', filters=filters, title='Xinet\'s dataset') as h5:
            for name in self:
                h5.create_group('/', name, title=name)
                if name in ['mnist', 'fashion_mnist']:
                    h5.create_array(
                        h5.root[name], 'trainX', self[name].train_data)
                    h5.create_array(
                        h5.root[name], 'trainY', self[name].train_label)
                    h5.create_array(
                        h5.root[name], 'testX', self[name].test_data)
                    h5.create_array(
                        h5.root[name], 'testY', self[name].test_label)
                elif name == 'cifar10':
                    h5.create_array(
                        h5.root[name], 'trainX', self._change(self[name].trainX))
                    h5.create_array(h5.root[name], 'trainY', self[name].trainY)
                    h5.create_array(
                        h5.root[name], 'testX', self._change(self[name].testX))
                    h5.create_array(h5.root[name], 'testY', self[name].testY)
                    h5.create_array(h5.root[name], 'label_names', np.array(
                        self[name].meta[b'label_names']))
                elif name == 'cifar100':
                    h5.create_array(
                        h5.root[name], 'trainX', self._change(self[name].trainX))
                    h5.create_array(
                        h5.root[name], 'testX', self._change(self[name].testX))
                    h5.create_array(
                        h5.root[name], 'train_coarse_labels', self[name].train_coarse_labels)
                    h5.create_array(
                        h5.root[name], 'test_coarse_labels', self[name].test_coarse_labels)
                    h5.create_array(
                        h5.root[name], 'train_fine_labels', self[name].train_fine_labels)
                    h5.create_array(
                        h5.root[name], 'test_fine_labels', self[name].test_fine_labels)
                    h5.create_array(h5.root[name], 'coarse_label_names', np.array(
                        self[name].meta[b'coarse_label_names']))
                    h5.create_array(h5.root[name], 'fine_label_names', np.array(
                        self[name].meta[b'fine_label_names']))
