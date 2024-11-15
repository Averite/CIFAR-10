import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False,
                                          transform=torchvision.transforms.ToTensor())

#dataset length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("train data size: {}".format(train_data_size))
print(f"test data size: {test_data_size}")

#Dataloader
train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

#nn
tudui = Tudui()

#loss
loss_fn = nn.CrossEntropyLoss()

##optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter(log_dir='./logs_train')

for i in range(epoch):
    print(f"------------第 {i+1}回の学習を開始--------------")

    tudui.train()
    for data in train_dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"学習回数：{total_train_step},Loss={loss.item()}")
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    tudui.eval()
    #test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy+accuracy
    print(f"全体的なloss:{total_test_loss}")
    print(f"全体的な正解率:{total_accuracy/test_data_size}")
    writer.add_scalar('test_loss', total_test_loss, total_test_step+1)

    torch.save(tudui.state_dict(), 'tudui_{}.path'.format(i))
    print("model saved")

writer.close()