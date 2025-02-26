import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=48, kernel_size=7, stride=1, padding=3
        )
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")

        self.lr_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(
            in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2
        )
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity="relu")

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        nn.init.kaiming_uniform_(self.conv5.weight, nonlinearity="relu")

        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=3
        )
        nn.init.kaiming_uniform_(self.conv6.weight, nonlinearity="relu")

        self.conv7 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=11, stride=1, padding=5
        )
        nn.init.kaiming_uniform_(self.conv7.weight, nonlinearity="relu")

        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=16, kernel_size=11, stride=1, padding=5
        )
        nn.init.kaiming_uniform_(self.conv8.weight, nonlinearity="relu")

        self.conv9 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=13, stride=1, padding=6
        )
        nn.init.kaiming_uniform_(self.conv9.weight, nonlinearity="relu")

        self.deconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=8, stride=4, bias=False
        )
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.0001)

    def forward(self, x):
        x = self.lr_norm(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.deconv(x)

        return x
