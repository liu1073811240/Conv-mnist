import torch.nn as nn
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()  # N, 16, 14, 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()  # N, 64, 7, 7
        )

        self.fc = nn.Linear(64*7*7, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        y = torch.reshape(y, [y.size(0), -1])
        y = self.fc(y)

        return y


if __name__ == '__main__':
    batch_size = 100
    save_params = "./net.params.pth"
    save_net = "./net.pth"

    train_data = datasets.MNIST("./mnist", True, transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("./mnist", False, transforms.ToTensor(), download=True)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0)

    plt.ion()
    a = []
    b = []
    net.train()

    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
            x = x.to(device)
            y = y.to(device)
            out = net(x)

            # 加正则化
            # L1 = 0
            # L2 = 0
            # for params in net.parameters():
            #     L1 += torch.sum(torch.abs(params))
            #     L2 += torch.sum(torch.pow(params, 2))
            # loss = loss_fn(out, y)
            # loss1 = loss + 0.001*L1
            # loss2 = loss + 0.001*L2
            # loss = 0.2*loss1 + 0.8*loss2

            loss = loss_fn(out, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 100 == 0:
                plt.clf()
                a.append(i + epoch*(len(train_data) / batch_size))
                b.append(loss.item())
                plt.plot(a, b)
                plt.pause(0.01)

                print("epoch:{}, loss:{:.3f}".format(epoch, loss.item()))

            torch.save(net.state_dict(), save_params)
            # torch.save(net, save_net)

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        y = torch.zeros(y.size(0), 10).scatter_(1, y.reshape(-1, 1), 1)
        x = x.to(device)
        y = y.to(device)

        out = net(x)
        loss = loss_fn(out, y)
        eval_loss += loss.item()*batch_size

        max_y = torch.argmax(y, 1)
        max_out = torch.argmax(out, 1)
        eval_acc += (max_out == max_y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)

    print(mean_loss, mean_acc)

















