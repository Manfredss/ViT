from datasets import MNIST
import matplotlib.pyplot as plt
from vit import ViT
import torch
import random
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MNIST()
model = ViT().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 随机选取一张图片
idx = random.randint(0, len(dataset))
img, label = dataset[idx]
print(f'Correct label: {label}')
plt.imshow(img.permute(1, 2, 0))
plt.show()

logits = model(img.unsqueeze(0).to(device))
print(f'Predicted label: {logits.argmax(dim=-1).item()}') 
