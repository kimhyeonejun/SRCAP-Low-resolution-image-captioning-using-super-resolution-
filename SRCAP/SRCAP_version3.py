#version 3
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import models, dataset, utils
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torch.utils.data import Dataset, DataLoader
import os

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import models, torch
from time import time
from tqdm import tqdm
import os
import warnings


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

if __name__ == "__main__":
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataDir='..'
    dataType='val2017'
    COCOdataset = dataset.train_dataset(utils.root, utils.training_dataset_json.format(dataDir,dataType))

    train_dataloader = DataLoader(dataset= COCOdataset, batch_size=10, shuffle=True, collate_fn= dataset.collate_fn, num_workers=10)
    img, anns = COCOdataset[0]
    plt.imshow(img.permute(1,2,0).numpy())

    warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
    warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    SR = torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')
    SR.eval()
    #SR.load_state_dict(torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')['model_state_dict']).to('cuda')
    #checkpoint = torch.load('/ssd1/HyunJun/SRGAN/modelmodel3.pt')

    VisionTransformer = utils.load("model_trained")
    G = models.SRCAP_version2_Generator(VisionTransformer= VisionTransformer).to('cuda').train()
    D = models.Discriminator().to('cuda').train()

    learning_rates = {
                'generator': 1e-5,
                'discriminator': 4e-4,
            }

    optimizerG = torch.optim.Adam(G.parameters(), lr = learning_rates['generator'])
    optimizerD = torch.optim.Adam(D.parameters(), lr = learning_rates['discriminator'])
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer   = optimizerG,
                                                step_size = utils.total_iters,
                                                gamma     = utils.gamma,
                                                )
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer = optimizerD,
                                                    step_size = utils.total_iters,
                                                    gamma     = utils.gamma)
    loop_train = tqdm(
                    enumerate(train_dataloader, 1),
                    total=len(train_dataloader),
                    desc="Train",
                    position=0,
                    leave=True,
                )

    for epoch in range(utils.num_epochs):
        start_time = time()    
        G_train_loss = []
        D_train_loss = []
        loss_avg_train = AverageMeter()
        for idx , (img_real, img, anns) in loop_train:
            img_real, img, anns = img_real.to('cuda'), img.to('cuda'), anns.to('cuda')

            #Generator
            optimizerG.zero_grad()
            fake_img, fake_anns = G(SR(img))
            print("fake_img", idx,": ", utils.load("tokenizer").batch_decode(anns[0],  skip_special_tokens=True))
            print("real_img", idx,": ", utils.load("tokenizer").batch_decode(fake_anns[0],  skip_special_tokens=True))
            #print("fake_anns", fake_anns)
            #print("anns", anns)
            G_loss = utils.G_loss(fake_anns/50237, anns/50237)
            G_loss.requires_grad_(True)
            G_loss.backward()
            optimizerG.step()
            G_train_loss.append(G_loss.item())
            #Discriminator
            
            optimizerD.zero_grad()
            discriminated_real_images = D(img.to('cuda'))
            discriminated_fake_images = D(fake_img.to('cuda'))
            D_loss = utils.GANLoss('vanilla', 'D', 'S')
            D_loss = D_loss.D_loss(discriminated_fake_images, discriminated_real_images)
            D_loss.requires_grad_(True)
            D_loss.backward()
            optimizerD.step()
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