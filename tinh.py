import json
import csv
import os
from module.detector.dataset import Taco
from module.mask import Mask
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np

ROOT_DIR = "module"
RCNN = Mask(
    PathToROOT=os.path.join(ROOT_DIR, "detector/models"),
    classMap=os.path.join(ROOT_DIR, "detector/taco_config/map_10.csv"),
    modelName="mask_02",
    pathToDataset=os.path.join(ROOT_DIR, "data"),
    splitnumber=2,
)
RCNN.detectBulkIMG("fileanhngoai2", "out2")
