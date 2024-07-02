import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, GPT2Config
from tqdm import tqdm
import json
import random
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

def load(option):
    #print(option)
    if(option == "model_trained"): return VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    elif(option == "image_processor"): return ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", do_rescale = False)
    elif(option == "tokenizer"): return GPT2TokenizerFast.from_pretrained("gpt2")
    elif(option == "LMHeadModel"): return GPT2LMHeadModel(GPT2Config())
    else: return
