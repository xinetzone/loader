"""
CASIA Online and Offline Chinese Handwriting Databases: 
http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
"""

import struct
import os
import numpy as np
import pandas as pd
import zipfile
import tables as tb

from ..dataset import Bunch, bunch2json, Bunch


class MPF:
    # MPF 文件的解码器
    def __init__(self, fp):
        self._fp = fp
        # 解码文件头
        header_size = struct.unpack('l', self._fp.read(4))[0]
        # 文件保存的形式，如 “MPF”
        self.code_format = self._fp.read(8).decode('ascii').rstrip('\x00')
        # 文本说明
        self.text = self._fp.read(header_size - 62).decode().rstrip('\x00')
        # 编码类型，如 “ASCII”, “GB”, etc
        self.code_type = self._fp.read(20).decode('latin-1').rstrip('\x00')
        # 编码长度
        self.code_length = struct.unpack('h', self._fp.read(2))[0]
        self.dtype = self._fp.read(20).decode('ascii').rstrip('\x00')
        if self.dtype == 'unsigned char':
            self.dtype = np.uint8
        else:
            self.dtype = np.dtype(self.dtype)
        # 样本数
        self.nrows = struct.unpack('l', self._fp.read(4))[0]
        # 特征的维度
        self.ndims = struct.unpack('l', self._fp.read(4))[0]

    def __iter__(self):
        m = self.code_length + self.ndims
        for i in range(0, m * self.nrows, m):
            # 样本的标签
            label = self._fp.read(self.code_length).decode('gb2312-80')
            # 样本的特征
            data = np.frombuffer(self._fp.read(self.ndims), self.dtype)
            yield data, label


def MPF2Bunch(set_name):
    # 将 MPF 转换为 bunch
    Z = zipfile.ZipFile(set_name)
    mb = Bunch()
    for name in Z.namelist():
        if name.endswith('.mpf'):
            writer_ = f"writer{os.path.splitext(name)[0].split('/')[1]}"
            with Z.open(name) as fp:
                mpf = MPF(fp)
                df = pd.DataFrame.from_records(
                    [(label, data) for data, label in mpf])
                df = pd.DataFrame(data=np.column_stack(df[1]).T, index=df[0])
                mb[writer_] = Bunch({"text": mpf.text, 'features': df})
    return mb


def bunch2hdf(root_dict, save_path):
    filters = tb.Filters(complevel=7, shuffle=False)  # 过滤信息，用于压缩文件
    h = tb.open_file(save_path, 'w', filters=filters, title='Xinet\'s dataset')
    for name in root_dict:  # 生成数据集"头"
        h.create_group('/', name=name, filters=filters)
        for writer_id in root_dict[name].keys():  # 生成写手
            h.create_group('/'+name, writer_id)
            h.create_array(f"/{name}/{writer_id}", 'text',
                           root_dict[name][writer_id]['text'].encode())
            features = root_dict[name][writer_id]['features']
            h.create_array(f"/{name}/{writer_id}", 'labels',
                           " ".join([l for l in features.index]).encode())
            h.create_array(f"/{name}/{writer_id}", 'features', features.values)
    h.close()  # 防止资源泄露


if __name__ == "__main__":
    root = 'E:/OCR/CASIA/data'  # CASI 数据集所在根目录
    feature_paths = {
        os.path.splitext(name)[0].replace('.', ''):
        f'{root}/{name}' for name in os.listdir(root) if '_' not in name}
    root_dict = Bunch({
        name: MPF2Bunch(feature_paths[name]) for name in feature_paths
    })

    hdf_path = 'E:/OCR/CASIA/datasets/features.h5'
    json_path = 'E:/OCR/CASIA/datasets/features.json'
    bunch2hdf(root_dict, hdf_path)
    bunch2json(root_dict, json_path)