from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # self.conv1 = nn.Conv2d(3,32,5,padding=2)
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32,32,5,padding=2)
        # self.maxpool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32,64,5,padding=2)
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(64*4*4,64)
        # self.linear2 = nn.Linear(64,10)
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
    #     x = self.conv1(x)
    #     x = self.maxpool1(x)
    #     x = self.conv2(x)
    #     x = self.maxpool2(x)
    #     x = self.conv3(x)
    #     x = self.maxpool3(x)
    #     x = self.flatten(x)
    #     x = self.linear1(x)
    #     x = self.linear2(x)
        return x



tudui = Tudui()
print(tudui)

input = torch.ones(64,3,32,32)
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(tudui,input)
writer.close()