from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "/Users/averite/Desktop/LearnPytorch/dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array=np.array(img_PIL)

writer.add_image("test",img_array,4,dataformats='HWC')
#y=x
# for i in range(100):
#     writer.add_scalar("y=2x",3*i,i)


writer.close()