import torch
import torchvision.transforms
from PIL import Image
from mpmath.identification import transforms
from torch import nn
from model import *

image_path = "./dataset/dog.png"

image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])

image = transform(image)


model = Tudui()


model.load_state_dict(torch.load('tudui_9.path'), strict=False)
print(model)

image = torch.reshape(image, (1,3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
#https://atmarkit.itmedia.co.jp/ait/articles/2006/10/news021.html