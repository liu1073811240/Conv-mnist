import torch.nn as nn
import torch
import thop


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2,
                               padding=1)

    def forward(self, x):
        y = self.conv1(x)
        print(y.shape)  # torch.Size([100, 16, 14, 14])

        return y


if __name__ == '__main__':
    data = torch.randn(100, 1, 28, 28)
    net = Net()
    net(data)

    flops, params = thop.profile_origin(net, (data, ))
    print(flops, params)  # 浮点数运算量，参数

    flops,  params = thop.clever_format((flops, params), format="%.2f")
    print(flops, params)



