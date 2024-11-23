import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs,targets = data
    print(f"Original image shape: {imgs.shape}")  # (64, 3, 32, 32)

    # 将整个批次展平，转换为一维
    output = torch.flatten(imgs)
    print(f"Flattened batch shape: {output.shape}")  # (196608,)

    # 将展平后的整个批次输入到模型
    output = tudui(output)
    print(f"Output shape: {output.shape}")  # (10,)