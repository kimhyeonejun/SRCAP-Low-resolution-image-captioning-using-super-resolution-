import os
import utils
from PIL import Image
import dataset
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.utils.rnn as rnn_utils
import torch
import warnings
import models
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

class train_dataset(Dataset): #Output : LR_img and annotations
    def __init__(self, root, annFile):
        super().__init__()
        self.root = root
        self.coco = COCO(os.path.join(root, annFile))
        self.whole_imgIds = self.coco.getImgIds()
        self.image_ids = []
        for idx in self.whole_imgIds:
	        annotations_ids = self.coco.getAnnIds(imgIds=idx)
	        if len(annotations_ids) != 0:
	            self.image_ids.append(idx)
        self.upscaling_factor = utils.upscaling_factor
        self.crop_size = utils.crop_size
    def __getitem__(self, index):
        imageId = self.coco.getImgIds(imgIds= self.image_ids[index])
        annId = self.coco.getAnnIds(imgIds= self.image_ids[index])
        if len(annId) == 0:
            annId = np.zeros((0, 5))
        img = dataset.lr_transform(coco = self.coco, root = self.root, imageId = imageId, crop_size = self.crop_size, upscaling_factor = self.upscaling_factor)
        k = cv2.bilateralFilter(img.permute(1,2,0).cpu().detach().numpy(), -1, 0.01, 5)
        img = torch.tensor(k)
        img = img.permute(2,0,1)
        ann = self.coco.loadAnns(annId)
        anns = []
        for i in range(len(ann)):
             k = utils.decoder(ann[i]['caption'])
             anns.append(k)
        anns = rnn_utils.pad_sequence([ann[0].clone().detach() for ann in anns], batch_first=True, padding_value=0)
        anns = anns.clone().detach()
        #print(anns)
        return img, anns[0]
    
    def __len__(self):
        return len(self.image_ids)
    
class train_dataset_version2(Dataset): #Output : HR_img, LR_img and annotations
    def __init__(self, root, annFile):
        super().__init__()
        self.root = root
        self.coco = COCO(os.path.join(root, annFile))
        self.whole_imgIds = self.coco.getImgIds()
        self.image_ids = []
        self.upscaling_factor = utils.upscaling_factor
        self.crop_size = utils.crop_size
        for idx in self.whole_imgIds:
	        annotations_ids = self.coco.getAnnIds(imgIds=idx)
	        if len(annotations_ids) != 0:
	            self.image_ids.append(idx)
                 
    def __getitem__(self, index):
        while True:
            anns = []
            imageId = self.coco.getImgIds(imgIds= self.image_ids[index])
            annId = self.coco.getAnnIds(imgIds= self.image_ids[index])
            ann = self.coco.loadAnns(annId)
            for i in range(len(ann)):
                anns.append(ann[i]['caption'])
            anns = " ".join(anns)
            anns = utils.decoder(anns)
            anns = rnn_utils.pad_sequence([anns], batch_first=True, padding_value=50257)
            anns = anns.view(-1).unsqueeze(0)
            if len(anns) != 0 and len(annId) != 0: break
            else: index = index + 1

        img_real, img = dataset.lr_transform(coco = self.coco, root = self.root, imageId = imageId, crop_size = 1, upscaling_factor = self.upscaling_factor)

        anns = anns.clone().detach()
        return img_real, img, anns[0]
    
    def __len__(self):
        return len(self.image_ids)