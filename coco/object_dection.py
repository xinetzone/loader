'''
COCO 数据集的目标检测结构
'''
from dataclasses import dataclass
from typing import List, Any

from .coco_meta import Info, License, Image


@dataclass
class Category:
    id: int
    name: str
    supercategory: str


@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    segmentation: Any  # RLE or [polygon]
    area: float
    bbox: Any  # [x, y, width, height]
    iscrowd: Any  # 0 or 1


@dataclass
class Dataset:
    info: Info
    licenses: List[License]
    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]
