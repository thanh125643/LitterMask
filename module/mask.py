import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import numpy as np
import json
import csv
import random
import colorsys
from imgaug import augmenters as iaa
from tqdm import tqdm

from module.detector.dataset import Taco
from module.detector import model as modellib
from module.detector.model import MaskRCNN
from module.detector.config import Config
from module.detector import visualize
from module.detector import utils
import cv2


from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class Mask:
    def __init__(
        self,
        PathToROOT="./detector/models",
        classMap="detector/taco_config/map_10.csv",
        modelName="mask_rcnn_taco_0100",
        pathToDataset="data",
        splitnumber=0,
    ):
        self.ROOT_DIR = PathToROOT
        # Path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        # Directory to save logs and model checkpoints
        self.DEFAULT_LOGS_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.pathCSV = classMap
        self.pathDataset = pathToDataset
        self.modelName = modelName
        self.splitnumber = splitnumber
        self.getclass()
        self.prepareDataset()
        self.model = MaskRCNN(
            mode="inference", config=self.config, model_dir=self.DEFAULT_LOGS_DIR
        )
        self.modelLoad()

    def getclass(self):
        self.class_map = {}
        self.map_to_one_class = {}
        with open(self.pathCSV) as csvfile:
            reader = csv.reader(csvfile)
            self.class_map = {row[0]: row[1] for row in reader}
            self.map_to_one_class = {c: "Litter" for c in self.class_map}

    def prepareDataset(self):
        self.dataset = Taco()
        self.taco = self.dataset.load_taco(
            self.pathDataset,
            self.splitnumber,
            "test",
            class_map=self.class_map,
            return_taco=True,
        )
        self.dataset.prepare()
        nr_classes = self.dataset.num_classes

        class TacoTestConfig(Config):
            NAME = "taco"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 10
            NUM_CLASSES = nr_classes
            USE_OBJECT_ZOOM = False

        self.config = TacoTestConfig()
        self.config.display()

    def modelLoad(self):
        _, model_path = self.model.get_last_checkpoint(self.modelName)
        self.model.load_weights(model_path, model_path, by_name=True)
        self.model.keras_model._make_predict_function()

    def resizeImg(self, img, maxDim, minDim):
        h, w = img.shape[:2]
        orgin = (w, h)
        scale = 1
        scale = max(1, minDim / min(h, w))

        image_max = max(h, w)
        if round(image_max * scale) > maxDim:
            scale = maxDim / image_max
        img = cv2.resize(img, (round(w * scale), round(h * scale)))

        h, w = img.shape[:2]
        delta_w = maxDim - w
        delta_h = maxDim - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0]
        )
        return img, orgin, (top, bottom, left, right), scale

    def randomColor(self, nIds):
        brightness = 0.7
        hsv = [(i / nIds, 1, brightness) for i in range(nIds)]
        colors = list(map(lambda c: np.multiply(colorsys.hsv_to_rgb(*c), 255), hsv))
        random.shuffle(colors)
        return colors

    def revertMask(self, rdata, orgin, pad, scale):
        if rdata["class_ids"].shape[0] == 0:
            return rdata
        h, w = rdata["masks"].shape[:2]
        maskfinal = []
        for i in range(rdata["masks"].shape[2]):
            mask = rdata["masks"][pad[0] : h - pad[1], pad[2] : w - pad[3], i]
            mask = cv2.resize(mask.astype(np.uint8), orgin)
            maskfinal.append(mask)
        if len(maskfinal) > 1:
            maskfinal = np.dstack(tuple(maskfinal))
        else:
            maskfinal = np.reshape(
                maskfinal[0], (maskfinal[0].shape[0], maskfinal[0].shape[1], 1)
            )
        rdata["masks"] = maskfinal
        for i in range(len(rdata["rois"])):
            y1, x1, y2, x2 = rdata["rois"][i]
            rdata["rois"][i] = (y1 - pad[0], x1 - pad[2], y2 - pad[0], x2 - pad[2])
            rdata["rois"][i] = np.divide(rdata["rois"][i], scale)
        return rdata

    def maskIMG(self, img, rdata, className):
        ncolor = self.randomColor(len(rdata["class_ids"]))
        for i in range(len(ncolor)):
            y1, x1, y2, x2 = rdata["rois"][i]
            color = ncolor[i].tolist()
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            score = rdata["scores"][i]
            label = className[rdata["class_ids"][i]]
            caption = "{} {:.3f}".format(label, score)
            img = cv2.putText(
                img,
                caption,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            img = visualize.apply_mask(img, rdata["masks"][:, :, i], color)

        return img

    def loadPicGT(self, input, output, ids):
        img = cv2.imread(input, cv2.IMREAD_COLOR)
        anns = self.taco.loadAnns(self.taco.getAnnIds([ids]))
        rois = []
        class_ids = []
        segmentation = []
        for i in anns:
            rois.append(i["bbox"])
            class_ids.append(i["category_id"])
            temp = np.array(
                [
                    np.array([[x, y]], dtype=int)
                    for x, y in zip(
                        i["segmentation"][0][::2], i["segmentation"][0][1::2]
                    )
                ]
            )
            segmentation.append(temp)
        overlay = img.copy()
        ncolor = self.randomColor(len(class_ids))
        for i in range(len(ncolor)):
            x, y, w, h = rois[i]
            color = ncolor[i].tolist()
            img = cv2.rectangle(
                img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2
            )

            label = self.dataset.class_names[class_ids[i]]
            caption = "{}".format(label)
            img = cv2.putText(
                img,
                caption,
                (int(x), int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            overlay = cv2.drawContours(overlay, [segmentation[i]], -1, color, -1)
        cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
        cv2.imwrite(output, img)

    def detectIMG(self, pathOfImage, OutputPath):
        img = cv2.imread(pathOfImage, cv2.IMREAD_COLOR)
        imgresult = img.copy()
        img, orgin, pad, scale = self.resizeImg(img, 1024, 800)
        r = self.model.detect([img], verbose=0)[0]
        if r["class_ids"].shape[0] > 0:
            r_fuse = utils.fuse_instances(r)
        else:
            r_fuse = r
        rdata = self.revertMask(r_fuse, orgin, pad, scale)
        img = self.maskIMG(imgresult, rdata, self.dataset.class_names)
        cv2.imwrite(OutputPath, img)
        return rdata

    def detectBulkIMG(self, InputPath, OutputPath):
        if not os.path.exists(InputPath):
            print("no path: " + InputPath)
        if not os.path.exists(OutputPath):
            os.mkdir(OutputPath)
        res = []
        with tqdm(os.listdir(InputPath)) as tq:
            for i in tq:
                if (os.path.splitext(i)[1]).lower() not in [".jpg", ".png"]:
                    continue
                pathImage = os.path.join(InputPath, i)
                OutputImage = os.path.join(OutputPath, i)
                tq.set_description(pathImage)
                img = cv2.imread(pathImage, cv2.IMREAD_COLOR)
                imgresult = img.copy()
                img, orgin, pad, scale = self.resizeImg(img, 1024, 800)
                r = self.model.detect([img], verbose=0)[0]
                if r["class_ids"].shape[0] > 0:
                    r_fuse = utils.fuse_instances(r)
                else:
                    r_fuse = r
                rdata = self.revertMask(r_fuse, orgin, pad, scale)
                img = self.maskIMG(imgresult, rdata, self.dataset.class_names)
                cv2.imwrite(OutputImage, img)
                res.append(rdata)
        return res

    def checkeval(self, outFolder, jsonOutput):
        jsonOut = []
        imgData = self.taco.imgs
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)
        pathPredict = os.path.join(outFolder, "Predicet")
        pathGroundTrust = os.path.join(outFolder, "GT")
        if not os.path.exists(pathPredict):
            os.mkdir(pathPredict)
        if not os.path.exists(pathGroundTrust):
            os.mkdir(pathGroundTrust)
        with tqdm(imgData.keys()) as pbar:
            for i in pbar:
                inputImg = os.path.join(self.pathDataset, imgData[i]["file_name"])
                pbar.set_description(inputImg)
                outputImg = os.path.join(
                    pathPredict,
                    imgData[i]["file_name"].split("/")[0]
                    + "_"
                    + os.path.basename(imgData[i]["file_name"]),
                )
                outputGT = os.path.join(
                    pathGroundTrust,
                    imgData[i]["file_name"].split("/")[0]
                    + "_"
                    + os.path.basename(imgData[i]["file_name"]),
                )
                self.loadPicGT(inputImg, outputGT, i)
                r = self.detectIMG(inputImg, outputImg)
                rclass = r["class_ids"].tolist()

                for a in range(len(r["class_ids"])):
                    rle = maskUtils.encode(np.asfortranarray(r["masks"][:, :, a]))
                    rle["counts"] = str(rle["counts"], "utf-8")
                    jsonOut.append(
                        {
                            "image_id": i,
                            "category_id": rclass[a],
                            "segmentation": rle,
                            "score": float(r["scores"][a]),
                        }
                    )
        with open(jsonOutput, "w") as f:
            json.dump(jsonOut, f)
        Pre = self.taco.loadRes(jsonOutput)
        result = COCOeval(self.taco, Pre)
        result.evaluate()
        result.accumulate()
        result.summarize()
