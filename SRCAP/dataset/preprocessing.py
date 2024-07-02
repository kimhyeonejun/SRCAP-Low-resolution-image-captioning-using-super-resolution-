from PIL import Image
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize, RandomCrop
import os
import cv2
import selectivesearch
import numpy as np
import utils
def random_crop(img, crop_size):
    img = Compose([RandomCrop(crop_size),
                   ToTensor()])
    return img

def hr_to_lr(coco, root, imageId, upscaling_factor):
    imageInfo = coco.loadImgs(imageId)[0]
    with open(os.path.join(root, 'train2017', imageInfo['file_name']), 'rb') as f:
        img = Image.open(f).convert('RGB')
    width, height = img.size
    transform = Compose([ 
        Resize((height // upscaling_factor, width // upscaling_factor), interpolation=Image.BICUBIC), 
        Resize((height, width), interpolation=Image.BILINEAR),
        Resize((64, 64)),
        ToTensor(),
        #Normalize(mean=[0.5, 0.5, 0.5],
        #          std=[0.1, 0.1, 0.1])
    ])
    transform_real = Compose([
        Resize((256, 256)),
        ToTensor()
    ])
    return transform_real(img), transform(img)

def lr_transform(coco, root, imageId, crop_size, upscaling_factor):
    if crop_size == 1:
        img_real, img = hr_to_lr(coco, root, imageId, upscaling_factor)
        return img_real, img
    else:
        _, img = hr_to_lr(coco, root, imageId, upscaling_factor)
        return img
def segmentation(image):
    regions, _ = selectivesearch.selective_search(image.cpu().detach().numpy(), sigma = 0.8, min_size = 100, scale = 100)
    region_proposals = regions[:, :, 3]
    segmentation_proposals = np.array([(region_proposals == k).astype(int) for k in range(round(np.max(region_proposals))+1)])
    #print("a", segmentation_proposals.shape)
    if len(segmentation_proposals) > utils.num_proposals : return segmentation_proposals[ : utils.num_proposals]
    else: return np.pad(segmentation_proposals, ((0, utils.num_proposals - len(segmentation_proposals)), (0, 0), (0, 0)), mode='constant', constant_values=0)

def image_preprocess(image):
    image_processor = utils.load("image_processor")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    return pixel_values