import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class TrashCanDataset(CocoDataset):

    CLASSES = ('rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells', 'animal_crab',
               'animal_eel', 'animal_etc', 'trash_clothing', 'trash_pipe', 'trash_bottle',
               'trash_bag', 'trash_snack_wrapper', 'trash_can', 'trash_cup', 'trash_container', 'trash_unknown_instance',
               'trash_branch', 'trash_wreckage', 'trash_tarp', 'trash_rope', 'trash_net')

class TrashCanMaterialDataset(CocoDataset):

    CLASSES = ('animal_crab', 'animal_eel', 'animal_etc', 'animal_fish', 'animal_shells', 'animal_starfish',
             'plant', 'rov', 'trash_etc', 'trash_fabric', 'trash_fishing_gear', 'trash_metal', 'trash_paper', 
             'trash_plastic', 'trash_rubber', 'trash_wood')