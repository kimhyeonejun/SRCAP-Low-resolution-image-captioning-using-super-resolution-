from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.image import imread
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

def load_image(coco, ids):
    imgIds = coco.getImgIds(imgIds = [ids])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    I = io.imread(img['coco_url'])
    return I

def print_image(dataDir, dataType, img, I):
    annFile = '/ssd1/HyunJun/SRCAP/annotations/captions_train2017.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)

    # load and display caption annotations
    annIds = coco_caps.getAnnIds(imgIds=img['id']);
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)
    plt.imshow(I); plt.axis('off'); plt.show()

def collate_fn(batch):
    """
    This is a function for collating a batch of variable-length sequences into a single tensor, which is
    useful for training a neural network with PyTorch.

    The input to this function is a batch of samples, each containing a source and target sequence. 
    The function extracts the source and target sequences from each sample, and then pads them to ensure
    that all sequences in the batch have the same length. This is necessary because PyTorch requires all
    inputs to a neural network to have the same shape.

    The function uses the PyTorch pad_sequence function to pad the sequences. pad_sequence is called with
    the batch_first=True argument to ensure that the batch dimension is the first dimension of the output
    tensor. The padding_value argument is set to 0 to pad with zeros.

    """
    _one = [item[0] for item in batch]
    _two = [item[1] for item in batch]
    #print(_two[0].shape)
    #print(_two[1].shape)
    #_thr = [item[2] for item in batch]
              
    _1 = torch.stack(_one, dim = 0)
    #_2 = torch.stack(_two)
    #print(batch[0][1].shape)
    #if _two:
        #print(_two[0].shape)
    #else:
        #print("FUCK")
    _2 = torch.nn.utils.rnn.pad_sequence(_two, batch_first=True)
    #print(_2.shape)
    return _1.contiguous(), _2.contiguous()

def collate_fn_version2(batch):
    """
    This is a function for collating a batch of variable-length sequences into a single tensor, which is
    useful for training a neural network with PyTorch.

    The input to this function is a batch of samples, each containing a source and target sequence. 
    The function extracts the source and target sequences from each sample, and then pads them to ensure
    that all sequences in the batch have the same length. This is necessary because PyTorch requires all
    inputs to a neural network to have the same shape.

    The function uses the PyTorch pad_sequence function to pad the sequences. pad_sequence is called with
    the batch_first=True argument to ensure that the batch dimension is the first dimension of the output
    tensor. The padding_value argument is set to 0 to pad with zeros.

    """
    _one = [item[0] for item in batch]
    _two = [item[1] for item in batch]
    _thr = [item[2] for item in batch]
              
    _1 = torch.stack(_one, dim = 0)
    _3 = torch.nn.utils.rnn.pad_sequence(_thr, batch_first=True)
    _2 = torch.stack(_two, dim = 0)
    return _1.contiguous(), _2.contiguous(), _3.contiguous()