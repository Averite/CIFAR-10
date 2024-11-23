import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])

output = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self,input):
        output = self.sigmoid(input)
        return output

tudui = Tudui()

writer=SummaryWriter("../logs_relu")

step =0

for data in dataloader:
    imgs, target = data
    writer.add_image("input", torchvision.utils.make_grid(imgs), global_step=step)

    output = tudui(imgs)

    writer.add_image("output", torchvision.utils.make_grid(output), global_step=step)
    step += 1

writer.close()