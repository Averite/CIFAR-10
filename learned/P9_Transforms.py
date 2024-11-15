from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》tensor数据类型
# 通过 transforms.Totensor 去看两个问题
# 1， transforms该被如何使用
# 2， 为什么我们需要Tensor数据类型

#绝对路径 /Users/averite/Desktop/LearnPytorch/dataset/train/ants_image/0013035.jpg
#相对路径 dataset/train/ants_image/0013035.jpg

img_path ="dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
