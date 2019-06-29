数据集的下载网址：[CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)
![][1]
中文申请书：

-  [CASIA-HWDB](http://www.nlpr.ia.ac.cn/databases/download/CASIA-HWDB-Chinese.pdf)
-  [CASIA-OLHWDB](http://www.nlpr.ia.ac.cn/databases/download/CASIA-OLHWDB-Chinese.pdf)


在申请书中介绍了数据集的基本情况：

&gt;  **CASIA-HWDB** 和 **CASIA-OLHWDB** 数据库由中科院自动化研究所在 2007-2010 年间收集， 均各自包含 1,020 人书写的脱机（联机）手写中文单字样本和手写文本， 用 Anoto 笔在点阵纸上书写后扫描、分割得到。

 1. **CASIA-HWDB** 手写单字样本分为三个数据库：*HWDB1.0~1.2*，手写文本也分为三个数据库： *HWDB2.0~2.2*。 
    - HWDB1.0~1.2 总共有 3,895,135 个手写单字样本，分属 7,356 类（7,185 个汉字和 171 个英文字母、数字、符号）。
    - HWDB2.0~2.2 总共有 5,091 页图像，分割为 52,230 个文本行和 1,349,414 个文字。所有文字和文本样本均存为灰度图像（背景已去除），按书写人序号分别存储。

 2. **CASIA-OLHWDB** 手写单字样本分为三个数据库：*OLHWDB1.0~1.2*，手写文本也分为三个数据库： *OLHWDB2.0~2.2*。 
    - OLHWDB1.0~1.2 总共有 3,912,017 个手写单字样本，分属 7,356 类（7,185 个汉字和 171 个英文字母、数字、符号）。   
    -     OLHWDB2.0~2.2 总共有 5,092 页手写文本，分割为 52,221 个文本行和 1,348,904 个文字。所有文字和文本样本均存为笔划坐标序列，按书写人序号分别存储。

&gt;  学术研究的用途包括：手写文档分割、字符识别、字符串识别、文档检索、书写人适应、书写人鉴别等。

我将 [Data Download](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) 下的数据集都下载到了 `root` 目录下：

```python
import os

root = 'E:/OCR/CASIA/'
os.listdir(root)
```




    ['1.0test-gb1.rar',
     '1.0test-GB1.zip',
     '1.0train-gb1.rar',
     '1.0train-GB1.zip',
     'competition-dgr.zip',
     'competition-gnt.zip',
     'competition_POT.zip',
     'Competition_ptts.zip',
     'HWDB1.0trn.zip',
     'HWDB1.0tst.zip',
     'HWDB1.1trn.zip',
     'HWDB1.1trn_gnt.zip',
     'HWDB1.1tst.zip',
     'HWDB1.1tst_gnt.zip',
     'mpf',
     'OLHWDB1.0trn.zip',
     'OLHWDB1.0tst.zip',
     'OLHWDB1.1trn.zip',
     'OLHWDB1.1trn_pot.zip',
     'OLHWDB1.1tst.zip',
     'OLHWDB1.1tst_pot.zip',
     'text']



我将其**特征数据**封装为 HDF5 文件，便于数据的处理，下面是我写的 API：


```python
import os
import sys
import zipfile, rarfile
import struct
import pandas as pd
import numpy as np
import tables as tb
import time


def getZ(filename):
    name, end = os.path.splitext(filename)
    if end == '.rar':
        Z = rarfile.RarFile(filename)
    elif end == '.zip':
        Z = zipfile.ZipFile(filename)
    return Z


class Bunch(dict):
    
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.__dict__ = self


class MPF:
    
    def __init__(self, fp):
        self.fp = fp
        header_size = struct.unpack('l', self.fp.read(4))[0]
        self.code_format = self.fp.read(8).decode('ascii').rstrip('\x00')
        self.text = self.fp.read(header_size - 62).decode().rstrip('\x00')
        self.code_type = self.fp.read(20).decode('latin-1').rstrip('\x00')
        self.code_length = struct.unpack('h', self.fp.read(2))[0]
        self.data_type = self.fp.read(20).decode('ascii').rstrip('\x00')
        self.nrows = struct.unpack('l', self.fp.read(4))[0]
        self.ndims = struct.unpack('l', self.fp.read(4))[0]
        
    def __iter__(self):
        m = self.code_length + self.ndims 
        for i in range(0, m * self.nrows, m):
            label = self.fp.read(self.code_length).decode('gbk')
            data = np.frombuffer(self.fp.read(self.ndims), np.uint8)
            yield data, label
            
            
class MPFBunch(Bunch):
    
    def __init__(self, root, set_name, *args, **kwds):
        super().__init__(*args, **kwds)
        filename, end = os.path.splitext(set_name)
        
        if 'HW' in filename and end == '.zip':
            if '_' not in filename:
                self.name = filename
                Z = getZ(f'{root}{set_name}')
                self._get_dataset(Z)
        else:
            #print(f'{filename}不是我们需要的文件！')
            pass
        
    def _get_dataset(self, Z):
        for name in Z.namelist():
            if name.endswith('.mpf'):
                writer_ = f"writer{os.path.splitext(name)[0].split('/')[1]}"
                
                with Z.open(name) as fp:
                    mpf = MPF(fp)
                    self.text = mpf.text
                    self.nrows = mpf.nrows
                    self.ndims = mpf.ndims
                    db = Bunch({label : data for data, label in iter(mpf)})
                    self[writer_] = pd.DataFrame.from_dict(db).T

                    
class BunchHDF5(Bunch):
    '''
    pd.read_hdf(path, wname) 可以直接获取 pandas 数据
    '''
    def __init__(self, mpf, *args, **kwds):
        super().__init__(*args, **kwds)
        
        if 'name' in mpf:
            print(f' {mpf.name} 写入进度条：')
            
        start = time.time()  
        for i, wname in enumerate(mpf.keys()):
            if wname.startswith('writer'):
                _dir = f'{root}mpf/'
                if not os.path.exists(_dir):
                    os.mkdir(_dir)
                self.path = f'{_dir}{mpf.name}.h5'
                mpf[wname].to_hdf(self.path, key = wname, complevel = 7)
                k = sys.getsizeof(mpf)     # mpf 的内存使用量
                print('-' * (1 + int((time.time() - start) / k)), end = '')
                
            if i == len(mpf.keys()) - 1:
                print('\n')

                
class XCASIA(Bunch):
    
    def __init__(self, root, *args, **kwds):
        super().__init__(*args, **kwds)
        self.paths = []
        print('开始写入磁盘')
        start = time.time()
        for filename in os.listdir(root):
            self.mpf = MPFBunch(root, filename)
            BunchHDF5(self.mpf)
        print(f'总共花费时间 {time.time() - start} 秒。')
```


```python
root = 'E:/OCR/CASIA/'
```


```python
%%time
xa = XCASIA(root)
```

    开始写入磁盘
     HWDB1.0trn 写入进度条：
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
     HWDB1.0tst 写入进度条：
    ------------------------------------------------------------------------------------
    
     HWDB1.1trn 写入进度条：
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
     HWDB1.1tst 写入进度条：
    ------------------------------------------------------------
    
     OLHWDB1.0trn 写入进度条：
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
     OLHWDB1.0tst 写入进度条：
    ------------------------------------------------------------------------------------
    
     OLHWDB1.1trn 写入进度条：
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
     OLHWDB1.1tst 写入进度条：
    ------------------------------------------------------------
    
    总共花费时间 618.7196779251099 秒。
    Wall time: 10min 18s


这个 API 封装了 以下数据集：
 ![图片描述][2]



```python
import os
import pandas as pd
import tables as tb

root = 'E:/OCR/CASIA/'
```


```python
os.listdir(f'{root}mpf/')
```




    ['HWDB1.0trn.h5',
     'HWDB1.0tst.h5',
     'HWDB1.1trn.h5',
     'HWDB1.1tst.h5',
     'OLHWDB1.0trn.h5',
     'OLHWDB1.0tst.h5',
     'OLHWDB1.1trn.h5',
     'OLHWDB1.1tst.h5']



下面我们来看一看，里面都是些什么：


```python
mpf_root = f'{root}mpf/'
for filename in os.listdir(mpf_root):
    with tb.open_file(f'{mpf_root}{filename}') as h5:
        print(h5.root)
    break
```

    / (RootGroup) ''
    


```python
mpf_root = f'{root}mpf/'
for filename in os.listdir(mpf_root):
    h5 = tb.open_file(f'{mpf_root}{filename}')
    break
```


```python
df = pd.read_hdf(f'{mpf_root}{filename}', key='writer001')
df
```




![图片描述][3]



```python
filename
```




    'HWDB1.0trn.h5'



上面是 'HWDB1.0trn' 数据集下写手 001 写的单字信息，总共有 3728 个汉字，每个汉字有 512 个特征。（由于该编译器不支持，故而只能截图。）


  [1]: //img.mukewang.com/5b39db5e0001a59507610115.png
  [2]: //img.mukewang.com/5b39e9e10001d41902100226.png
  [3]: //img.mukewang.com/5b39fa7b000132ab08160652.png

更多内容见：[datasetsome](https://dataloaderx.github.io/datasetsome/)
