from lib2to3.pgen2.tokenize import generate_tokens
import torch
import torch.nn as nn
import skimage.data as io
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from    torchvision.models          import  resnet50, ResNet50_Weights
import selectivesearch

class ResNetFeatureExtractor_proposedStructuredPooling(nn.Module):
    
    def __init__(self, cut, resize, resnet_OutPut, decoder_hiddenState, fine_tune_resnet):

        super(ResNetFeatureExtractor_proposedStructuredPooling, self).__init__()

        self.resize_ROI = resize
        # Load the pre-trained ResNet50 model
        self.resnet  = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:cut])

        self.fc1 = nn.Linear(resnet_OutPut, decoder_hiddenState)
        self.fc2 = nn.Linear(resnet_OutPut, decoder_hiddenState)

        self.resize_transform = Resize((self.resize_ROI,self.resize_ROI),antialias=True)

        if fine_tune_resnet:
            for name, param in self.resnet.named_parameters():
                if name.startswith('1') or name.startswith('2') or name.startswith('3'):
                    param.requires_grad = True
        else :
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self,
                Images,
                Masks,
                ):
        self.resnet.eval()
        #print(Images.shape)
        Masks = torch.from_numpy(Masks)
        #print(Masks.unsqueeze(0).shape)
        F             = self.resnet(Images.float())
        #print(F.shape)
        R_prime       = self.resize_transform(Masks.float()).to('cuda')
        #print(R_prime.shape)
        #plt.imshow(R_prime[0], cmap="gray")
        S = torch.matmul(R_prime.unsqueeze(2), F.unsqueeze(1)).sum(dim=-1).sum(dim=-1)/(self.resize_ROI*self.resize_ROI)
        #print(S.shape)

        return S

    def check(self,):
        #Prints the requires_grad attribute for each parameter in the ResNet model.
        for name, param in self.resnet.named_parameters():
            return print(name, param.requires_grad)