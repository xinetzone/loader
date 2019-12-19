import numpy as np
import tables as tb


class XDecode:
    def __init__(self, root='../datasome/X.h5'):
        '''解析数据 X
        使用完该实例记得关闭 self.h5.close()
        
        实例属性
        ========
        members 即 ['mnist', 'fashion_mnist', 'cifar100', 'cifar10']
        '''
        self.h5 = tb.open_file(root)
        self.members = self.h5.root.__members__

    def summary(self):
        '''打印 X 数据的摘要信息'''
        print(self.h5)

    def get_members(self, name):
        '''获得给定 name 的成员名称列表
        
        参数
        =======
        name in ['mnist', 'fashion_mnist', 'cifar100', 'cifar10']
        '''
        return self.h5.root[name].__members__

    def get_mnist_members(self):
        return self.h5.root.mnist.__members__

    def get_trainX(self, name):
        '''获得给定 name 的 trainX
        
        参数
        =======
        name in ['mnist', 'fashion_mnist', 'cifar100', 'cifar10']
        '''
        return self.h5.get_node(f'/{name}', 'trainX')

    def get_testX(self, name):
        '''获得给定 name 的 testX
        
        参数
        =======
        name in ['mnist', 'fashion_mnist', 'cifar100', 'cifar10']
        '''
        return self.h5.get_node(f'/{name}', 'testX')

    def get_trainY(self, name):
        '''获得给定 name 的 trainY
        
        参数
        =======
        name in ['mnist', 'fashion_mnist', 'cifar10']
        '''
        return self.h5.get_node(f'/{name}', 'trainY')

    def get_testY(self, name):
        '''获得给定 name 的 testY
        
        参数
        =======
        name in ['mnist', 'fashion_mnist', 'cifar10']
        '''
        return self.h5.get_node(f'/{name}', 'testY')

    def get_train_coarse_labels(self):
        '''获得 Cifar100 训练集的粗标签'''
        return self.h5.get_node('/cifar100', 'train_coarse_labels')

    def get_train_fine_labels(self):
        '''获得 Cifar100 训练集的细标签'''
        return self.h5.get_node('/cifar100', 'train_fine_labels')

    def get_test_coarse_labels(self):
        '''获得 Cifar100 测试集的粗标签'''
        return self.h5.get_node('/cifar100', 'test_coarse_labels')

    def get_test_fine_labels(self):
        '''获得 Cifar100 的测试集细标签'''
        return self.h5.get_node('/cifar100', 'test_fine_labels')

    def get_coarse_label_names(self):
        '''获得 Cifar100 测试集的粗标签的名称'''
        label_names = self.h5.get_node('/cifar100', 'coarse_label_names')
        return np.asanyarray(label_names, "U")

    def get_fine_label_names(self):
        '''获得 Cifar100 的测试集细标签的名称'''
        label_names = self.h5.get_node('/cifar100', 'fine_label_names')
        return np.asanyarray(label_names, "U")

    def get_label_names(self, name):
        '''获得给定 name 的标签名称
        
        参数
        ======
        name in ['mnist', 'fashion_mnist', 'cifar10']
        '''
        if name == 'cifar10':
            label_names = self.h5.get_node('/cifar10', 'label_names')
            return np.asanyarray(label_names, "U")
        elif name == 'fashion_mnist':
            return [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
        elif name == 'mnist':
            return np.arange(10)
