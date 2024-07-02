import  torch
import  utils
import  models
import  dataset
import  os ,tqdm
import  numpy                               as      np
import  matplotlib.pyplot                   as      plt
#import  torchvision.models                  as      models
import  torchvision.transforms              as      transforms
from    torch.utils.data                    import  DataLoader
from    nltk.translate.bleu_score           import  sentence_bleu, SmoothingFunction
import dataset
from torch.utils.data import Subset
import warnings
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


#directory_path      = r"/ssd1/HyunJun/SRCAP/models/training/LR=0.0001 d_out_mean=True batch_sizet=10 N_ROI=10 Fine_tuning=True"
#model               = r"LR=0.0001 d_out_mean=True batch_sizet=10 N_ROI=10 Fine_tuning=True.pt"
directory_path      = r"/ssd1/HyunJun/SRCAP/models/testing/LR=5e-05 Upscaling_factor= 12 d_out_mean=True batch_sizet=10 N_ROI=10 Fine_tuning=True"
model               = r"LR=5e-05 Upscaling_factor= 12 d_out_mean=True batch_sizet=10 N_ROI=10 Fine_tuning=True.pt"
chencherry = SmoothingFunction()
dataDir='..'
dataType='test2017'
COCOdataset = dataset.train_dataset_version2(utils.root, utils.training_dataset_json.format(dataDir,dataType))
indices = list(range(100))
COCOdataset = Subset(COCOdataset, indices)
test_dataloader = DataLoader(dataset= COCOdataset, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn_version2, num_workers= 15)

Tester = models.SRCAP_version2_Generator().to('cuda')
Tester.load_state_dict(torch.load(os.path.join(directory_path,model)))
Tester.eval()
scores          = [0,0,0,0]
count_failed    = [0,0,0,0]
count_sussec    = [0,0,0,0]
senteses = []
senteses_groundTruth = []

loop = tqdm.tqdm(
                            enumerate(test_dataloader, 1),
                            total=len(test_dataloader),
                            desc="BLEU",
                            position=0,
                            leave=True
                        )

with torch.inference_mode():
    for batch_idx, (img_real, img, anns) in loop:
        image_real = img_real.to('cuda')
        img = img.to('cuda')
        anns = anns.to('cuda')
#img : low-resolution image, img_real : high-resolution image, fake_img : super-resolved low-resolution image, real_anns : label annotations, fake_anns: annotations generated from super-resolved image, real_anns: annotations generated from high resolution image
        for index in range(utils.batch_size):
            fake_img, fake_anns = Tester(img, "Generator")
            real_anns = Tester.process_and_tokenizer(img).to('cuda')
            real_anns_tokenizer = utils.load("tokenizer").batch_decode(real_anns[index],  skip_special_tokens=True)
            anns_tokenizer = utils.load("tokenizer").batch_decode(anns[index], skip_special_tokens =True)
            fake_anns_tokenizer = utils.load("tokenizer").batch_decode(fake_anns[index], skip_special_tokens = True)
                
            try:
                scores[0] += sentence_bleu(["".join(anns_tokenizer),], "".join(real_anns_tokenizer) , weights = (1,0,0,0), smoothing_function=chencherry.method4)
                count_sussec[0] +=1
            except:
                count_failed[0] += 1
                
            try:
                scores[1] += sentence_bleu(["".join(anns_tokenizer),], "".join(real_anns_tokenizer), weights = (0.5,0.5,0,0), smoothing_function=chencherry.method4)
                count_sussec[1] +=1
            except:
                count_failed[1] += 1
            
            try:
                scores[2] += sentence_bleu(["".join(anns_tokenizer),], "".join(real_anns_tokenizer),weights = (0.333,0.333,0.334,0),smoothing_function=chencherry.method4)
                count_sussec[2] +=1
            except:
                count_failed[2] += 1
                
            try:
                scores[3] += sentence_bleu(["".join(anns_tokenizer),], "".join(real_anns_tokenizer),weights = (0.25,0.25,0.25,0.25),smoothing_function=chencherry.method4)
                count_sussec[3] +=1
            except:
                count_failed[3] += 1

            loop.set_description(f"iteration : {batch_idx}")
            loop.set_postfix(
                BLEU_1 = f"{(np.array(scores)/np.array(count_sussec))[0]:.4f}",
                BLEU_2 = f"{(np.array(scores)/np.array(count_sussec))[1]:.4f}",
                BLEU_3 = f"{(np.array(scores)/np.array(count_sussec))[2]:.4f}",
                BLEU_4 = f"{(np.array(scores)/np.array(count_sussec))[3]:.4f}",
                PSNR = f"{utils.psnr(fake_img, image_real): 4f}",
                refresh=True,
            )

    print(np.array(scores)/np.array(count_sussec))