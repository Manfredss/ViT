import torch
import os
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from vit import ViT
from datasets import MNIST
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
dataset = MNIST()
model = ViT().to(device)

try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# 设置训练参数
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch = 100
batchSize = 64
dataloader = DataLoader(dataset, batchSize, shuffle=True)

iterNum = 0
for iter in tqdm(range(epoch), desc='Training', ncols=75):
    for imgs, labels in dataloader:
        logits = model(imgs.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iterNum % 1000 == 0:
            print(f'>> epoch: {iter}, iter: {iterNum}, loss: {loss}')
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', f'model.pth')
        iterNum += 1