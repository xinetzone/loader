'''
COCO 数据集的通用结构
'''
from dataclasses import dataclass
import time


@dataclass
class Info:
    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created: time.asctime()  # 数据创建日期


@dataclass
class License:
    id: int
    name: str
    url: str


@dataclass
class Image:
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flickr_url: str
    coco_url: str
    date_captured: time.asctime()  # 图片标注时间
