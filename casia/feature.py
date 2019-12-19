"""
CASIA Online and Offline Chinese Handwriting Databases: 
http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
"""

import struct
import numpy as np
from pandas import DataFrame
from zipfile import ZipFile
import tables as tb


class MPFDecoder:
    def __init__(self, fp):
        '''MPF 文件的解码器

        参数
        ======
        fp: 可以是 open(mpf_name) 或者 zipfile.ZipFile(zip_name).open(mpf_name)
        处理任务完成务必关闭 fp

        实例属性
        ==========
        code_format: 文件保存的形式，这里是 MPF
        text: 数据特征的文本说明
        code_type: 编码类型，如 "ASCII"，GB","GB4" 等
        code_length: 编码长度
        dtype: 数据类型
        nrows: 样本个数
        ndims: 特征向量的长度
        dataset: mpf 解码后的数据 Frame
        '''
        self._decode(fp)
        assert (self.nrows, self.ndims) == self.dataset.shape, "数据不匹配"

    def _head(self, mpf):
        # 解码文件头
        header_size = struct.unpack('l', mpf.read(4))[0]
        # 文件保存的形式，如 “MPF”
        self.code_format = mpf.read(8).decode('ascii').rstrip('\x00')
        # 文本说明
        self.text = mpf.read(header_size - 62).decode().rstrip('\x00')
        # 编码类型，如 "ASCII"，GB","GB4"，etc
        code_type = mpf.read(20)
        self.code_type = code_type[:code_type.find(b'\x00')].decode('ascii')
        # 编码长度
        self.code_length = struct.unpack('h', mpf.read(2))[0]
        # 数据类型
        dtype = mpf.read(20).decode('ascii').rstrip('\x00')
        self.dtype = 'uint8' if dtype == 'unsigned char' else dtype
        # 样本数
        self.nrows = struct.unpack('l', mpf.read(4))[0]
        # 特征向量的维数长度
        self.ndims = struct.unpack('l', mpf.read(4))[0]

    def _get_feature(self, mpf):
        '''获取样本的特征向量'''
        feature = np.frombuffer(mpf.read(self.ndims), self.dtype)
        return feature

    def _get_label(self, mpf):
        '''获取样本的标签'''
        label = mpf.read(self.code_length).decode('gb18030')
        return label

    def _decode(self, mpf):
        '''解码 MPF 数据'''
        self._head(mpf)
        dataset = [[self._get_label(mpf), self._get_feature(mpf)]
                   for _ in range(self.nrows)]
        dataset = DataFrame.from_records(dataset)
        self.dataset = DataFrame(data=np.column_stack(
            dataset[1]).T, index=dataset[0])


def zipfile2bunch(zip_name):
    '''将 zip_name 转换为 bunch'''
    with ZipFile(zip_name) as Z:
        mb = {}  # 打包 mpf 文件
        for name in Z.namelist():
            if name.endswith('.mpf'):
                with Z.open(name) as mpf:
                    decoder = MPFDecoder(mpf)
                    mb[name] = {"text": decoder.text,
                                'dataset': decoder.dataset}
        return mb


def bunch2hdf(bunch, save_path):
    '''将 bunch 转换为 HDF5'''
    filters = tb.Filters(complevel=7, shuffle=False)  # 过滤信息，用于压缩文件
    h = tb.open_file(save_path, 'w', filters=filters, title='Xinet\'s dataset')
    for name in bunch:  # 生成数据集"头"
        _name = name.replace('/', '__')
        _name = _name.replace('.', '_')
        h.create_group('/', name=_name, filters=filters)
        h.create_array(f"/{_name}", 'text',
                       bunch[name]['text'].encode())
        features = bunch[name]['dataset']
        h.create_array(f"/{_name}", 'labels',
                       " ".join([l for l in features.index]).encode())
        h.create_array(f"/{_name}", 'features', features.values)
    h.close()  # 防止资源泄露


class CASIAFeature:
    def __init__(self, hdf_path):
        '''casia 数据 MPF 特征处理工具'''
        self.h5 = tb.open_file(hdf_path)

    def _features(self, mpf):
        '''获取 MPF 的特征矩阵'''
        return mpf.features[:]

    def _labels(self, mpf):
        '''获取 MPF 的标签数组'''
        labels_str = mpf.labels.read().decode()
        return np.array(labels_str.split(' '))

    def __iter__(self):
        '''返回 (features, labels)'''
        for mpf in self.h5.iter_nodes('/'):
            yield self._features(mpf), self._labels(mpf)