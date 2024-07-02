import dataset, models
import numpy as np
import utils
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")
warnings.filterwarnings('ignore', message=r'.*attention_mask.*', category=UserWarning)

class SRCAP_version1(nn.Module):
    def __init__(self):
        super(SRCAP_version1, self).__init__()
        self.SR = torch.load('/ssd1/HyunJun/SRGAN/model/model3.pt')
        self.resnet = models.ResNetFeatureExtractor_proposedStructuredPooling(cut = -2, resize = 8, resnet_OutPut = 2048, decoder_hiddenState = 512, fine_tune_resnet = False)
        self.ProposalClassifier = models.ProposalClassifier(num_proposals=utils.num_proposals)
        self.model = utils.load("LMHeadModel")
    
    def forward(self, I):
        I = self.SR(I)
        I_s = I.permute(0, 2, 3, 1)
        img_proposals = []
        for idx in range(len(I)):
            img_proposal = dataset.segmentation(I_s[idx])
            img_proposals.append(img_proposal)
        mask = np.array(img_proposals)
        S = self.resnet(I, mask)   
        output = self.ProposalClassifier(S) * 50256
        model = self.model
        token = model.generate(output.long(), max_length=10, num_return_sequences=1, pad_token_id= 50256)
        return I, token

class SRCAP_version2_Generator(nn.Module):
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SRCAP_version2_Generator, cls).__new__(cls, *args, **kwargs)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        super(SRCAP_version2_Generator, self).__init__()
        self.SR = models.SRResNet()
        self.image_preprocess = dataset.image_preprocess
        self.tokenizer = utils.tokenizer_process
        VisionTransformer = utils.load("model_trained").to('cuda')
        if utils.fine_tuning:
            for param in VisionTransformer.parameters():
                param.requires_grad = True
            self.VisionTransformer = VisionTransformer.train()
        else:
            for param in VisionTransformer.parameters():
                param.requires_grad = False
            self.VisionTransformer = VisionTransformer.eval()
        self.__initialized = True

    def process_and_tokenizer(self, img):
        I_p = img.permute(0, 2, 3, 1)
        I_q = torch.clip(I_p, 0, 1)
        pixel_values = self.image_preprocess(I_q)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        caption_results = self.VisionTransformer.generate(pixel_values, max_new_tokens = 30,  pad_token_id= 50256)
        caption_results = self.tokenizer(caption_results)
        caption_results = torch.tensor(np.array(caption_results)).to('cuda')
        return caption_results

    def forward(self, I, model):
        if model == "Generator":
            img = self.SR(I)
            caption_results = self.process_and_tokenizer(img)
            return img, caption_results
        else:
            img = self.SR(I)
            return img