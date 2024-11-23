from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_image/0013035.jpg")
print(img)

#Totensor的使用
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)


#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 2],[1,1,1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalization", img_norm,3)

#Resize
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)#img_resize -> PIL
img_resize = trans_toTensor(img_resize)#img_resize -> tensor
writer.add_image("Resize", img_resize, 3)

#Compose-resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2, 3)



writer.close()