# -*- coding: utf-8 -*-
"""SRCAP_version2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FuWWmz0rdWlxmePf2iHn3YQEC3AsA1x9
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import models, dataset, utils
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torch.utils.data import Dataset, DataLoader
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

from dataset import train_dataset
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import dataset
from tqdm import tqdm
from torch.utils.data import Subset

#pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='..'
dataType='val2017'
COCOdataset = dataset.train_dataset_version2(utils.root, utils.training_dataset_json.format(dataDir,dataType))
train_dataloader = DataLoader(dataset= COCOdataset, batch_size=utils.batch_size, shuffle=True, collate_fn= dataset.collate_fn_version2, num_workers= 15)
img_real, img, anns = COCOdataset[0]
plt.subplot(1,2,1)
plt.imshow(img_real.permute(1,2,0).numpy())
plt.subplot(1,2,2)
plt.imshow(img.permute(1,2,0).numpy())

print(img)

import torch
#SR = models.SRResNet()
SR = torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')
VisionTransformer = utils.load("model_trained")
#G = models.SRCAP_version2_Generator(SR = SR, VisionTransformer= VisionTransformer).to('cuda').train()
#fake_img, fake_anns = G(img.unsqueeze(0).to('cuda'))
#print("Keys in loaded state_dict:", SR.keys())
#SR.load_state_dict(state_dict)
#SR.eval()
image = SR(img.unsqueeze(0).to('cuda')).to('cuda')
image[0] = image[0]
print(image[0])
print(img_real)
plt.imshow(image[0].permute(1,2,0).cpu().detach().numpy())
print(utils.psnr(image.cpu().detach(), img_real))

#import utils
#VisionTransformer = utils.load("model_trained")

#from PIL import Image
#image = image.cpu().detach().numpy()
#image = np.clip(image, 0, 1)
#pixel_values = dataset.image_preprocess(image)
#print(pixel_values)

#plt.imshow(pixel_values[0].permute(1,2,0))
torch.cuda.empty_cache()

import models, torch
from time import time
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#SR = torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')
#SR.eval()
#SR.load_state_dict(torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')['model_state_dict']).to('cuda')
#checkpoint = torch.load('/ssd1/HyunJun/SRGAN/modelmodel3.pt')

#VisionTransformer = utils.load("model_trained")
G = models.SRCAP_version2_Generator().to('cuda').train()
D = models.Discriminator().to('cuda').train()

learning_rates = {
            'generator': utils.generator_learning_rate,
            'discriminator': utils.discriminator_learning_rate,
        }

optimizerG = torch.optim.Adam(G.parameters(), lr = learning_rates['generator'])
optimizerD = torch.optim.Adam(D.parameters(), lr = learning_rates['discriminator'])
lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer   = optimizerG,
                                            step_size = utils.total_iters,
                                            gamma     = utils.gamma,
                                            )
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer = optimizerD,
                                            step_size = utils.total_iters,
                                            gamma     = utils.gamma
                                            )


for epoch in range(utils.num_epochs):
    loop_train = tqdm(
                enumerate(train_dataloader, 1),
                total=len(train_dataloader),
                desc="Train",
                position=0,
                leave=True,
            )
    G_train_loss = []
    D_train_loss = []
    loss_avg_train = AverageMeter()
    for idx , (img_real, img, anns) in loop_train:
        with torch.autograd.set_detect_anomaly(True):
            img_real, img, anns = img_real.to('cuda'), img.to('cuda'), anns.to('cuda')

            #Generator
            optimizerG.zero_grad()
            fake_img, fake_anns = G(img, "Generator")
            #real_anns = G.process_and_tokenizer(img_real).to('cuda')
            real_anns = anns
            G_loss = utils.G_loss(fake_anns, real_anns, fake_img, img_real, D(fake_img)).to('cuda')
            G_loss.requires_grad_(True)
            G_loss.backward(retain_graph= True)
            optimizerG.step()
            lr_scheduler1.step()
            G_train_loss.append(G_loss.item())

            #Discriminator
            optimizerD.zero_grad()
            fake_img = G(img, "Discriminator")
            discriminated_real_images = D(img_real)
            discriminated_fake_images = D(fake_img.to('cuda'))
            D_loss = utils.GANLoss('vanilla', 'D', 'S')
            D_loss = D_loss.D_loss(discriminated_fake_images, discriminated_real_images)
            D_loss.requires_grad_(True)
            D_loss.backward()
            optimizerD.step()
            lr_scheduler2.step()
            D_train_loss.append(D_loss.item())
            test_psnr = utils.psnr(img_real, fake_img)
            loss_avg_train.update(G_loss.item(), anns.shape[0])
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                    G_loss_batch="{:.4f}".format(G_loss.detach().item()),
                    D_loss_batch="{:.4f}".format(D_loss.detach().item()),
                    psnr="{:.4f}".format(test_psnr),
                    refresh=True,
                )
    print(f'[{epoch + 1}, {utils.num_epochs:5d}] G_loss : {torch.tensor(G_train_loss).mean() : 3f} D_loss : {torch.tensor(D_train_loss).mean() : 3f}, D(x) : {discriminated_real_images.mean().item() : 3f}, D(G(z)) : {discriminated_fake_images.mean().item() : 3f}')

plt.axis('off')
plt.imshow(img_real[2].permute(1,2,0).cpu().detach().numpy())

plt.axis('off')
plt.imshow(img[2].permute(1,2,0).cpu().detach().numpy())

plt.axis('off')
plt.imshow(fake_img[2].permute(1,2,0).cpu().detach().numpy())

print(utils.load("tokenizer").batch_decode(anns[2],  skip_special_tokens=True)) #Original img caption

real_anns = G.process_and_tokenizer(img_real).to('cuda')
print(utils.load("tokenizer").batch_decode(real_anns[2],  skip_special_tokens=True)) #HR img caption

real_anns = G.process_and_tokenizer(img).to('cuda')
print(utils.load("tokenizer").batch_decode(real_anns[2],  skip_special_tokens=True)) #LR img caption

print(utils.load("tokenizer").batch_decode(fake_anns[2],  skip_special_tokens=True)) #SR img caption

model_name, folder_path = utils.make_dir()
utils.save_model(folder_path, model_name, G)