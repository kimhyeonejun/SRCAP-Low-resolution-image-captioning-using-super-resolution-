import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pycocotools.coco import COCO

def print_image(coco, ids, dataDir, dataType):
    coco_url = '/ssd1/HyunJun/SRCAP/annotations/captions_test2017.json'.format(dataDir,dataType)
    imgIds = coco.getImgIds(imgIds = [ids])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    I = io.imread(img['coco_url'])
    return I

def print_caption(dataDir, dataType, img, I):
    annFile = '/ssd1/HyunJun/SRCAP/annotations/captions_test2017.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)

    # load and display caption annotations
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)
    plt.imshow(I); plt.axis('off'); plt.show()
