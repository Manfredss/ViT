import torch
from torch import nn
from torch.nn import functional as F


class ViT(nn.Module):
    def __init__(self, embSize=16):
        super(ViT, self).__init__()
        # patch 大小是 4x4
        self.patchSize = 4
        self.patchNum = 28 // self.patchSize

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=self.patchSize**2,
                              kernel_size=self.patchSize,
                              padding=0,
                              stride=self.patchSize)
        # 对 patch 进行 embedding
        self.patchEmb = nn.Linear(in_features=self.patchSize**2,
                                  out_features=embSize)
        # 分类头输入
        self.clsToken = nn.Parameter(torch.rand(1, 1, embSize))
        # 位置编码
        self.posEmb = nn.Parameter(torch.rand(1, self.patchNum**2+1, embSize))
        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embSize,
                                                                                    nhead=2,
                                                                                    batch_first=True),
                                                        num_layers=3)
        # 分类头
        self.clsFC = nn.Linear(in_features=embSize,
                               out_features=10)
        
    def forward(self, x):
        # 原始大小 [batch, channel=1, height=28, width=28]
        # 卷积后大小 [batch, channel=16, height=7, width=7]
        x = self.conv(x)
        # 展平 [batchSize, channel=16, seqLen=49]
        x = x.view(x.size(0), x.size(1), self.patchNum**2)
        # [batchSize, seqLen=49, channel=16]
        x = x.permute(0, 2, 1)
        # patch 大小 [batchSize, seqLen=49, embSize]
        x = self.patchEmb(x)
        # [batchSize, 1, embSize]
        clsToken = self.clsToken.expand(x.size(0), 1, x.size(2))
        x = torch.cat((clsToken, x), dim=1)
        x = x + self.posEmb
        out = self.transformerEncoder(x)
        return self.clsFC(out[:, 0, :])
    

if __name__ == '__main__':
    model = ViT()
    x = torch.rand(5, 1, 28, 28)
    out = model(x)
    print(out.shape)
