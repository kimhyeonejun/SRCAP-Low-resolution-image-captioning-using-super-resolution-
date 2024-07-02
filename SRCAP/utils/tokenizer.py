import utils
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

tokenizer = utils.load("tokenizer")
def tokenizer_process(output):
    #model = utils.load("LMHeadModel")
    tokenizer = utils.load("tokenizer")
    encoded_fake_anns = []
    for i in range(len(output)):
        #print("True")
        #print(output[i].tolist())
        if i <= utils.batch_size:
            output_text = tokenizer.batch_decode(output[i], skip_special_tokens=True)
        else:
            return encoded_fake_anns 
        output_text = ''.join(output_text)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoded_fake_ann = tokenizer.encode(output_text, padding='max_length', max_length=utils.max_length, truncation=True, return_tensors='pt')
        #print(encoded_fake_ann.shape)
        encoded_fake_ann = encoded_fake_ann.cpu().detach().numpy()
        encoded_fake_anns.append(encoded_fake_ann[0])
    return encoded_fake_anns

def decoder(target_text):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    target_ids = tokenizer.encode(target_text, padding='max_length', max_length=utils.max_length, truncation=True, return_tensors='pt')
    #print(target_ids)
    return target_ids 