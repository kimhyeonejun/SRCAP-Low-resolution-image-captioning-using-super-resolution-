import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDiscriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False,
                 dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        #self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        inp = inp.view(2, -1)
        print(inp.shape)
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred
    
    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        
        inp = inp.long()
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred
     
class MaliGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.25):
        dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        super(MaliGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                        dropout)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, stride = 1, padding= 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, stride = 2, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3, stride= 2, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, stride= 2, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 512, out_channels= 512, kernel_size= 3, stride= 2, padding= 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)    
        )
    def forward(self, x):
        #x = self.net(x)
        batch_size = x.size()[0]
        return torch.sigmoid(self.net(x).view(batch_size))
