""""
if you want to use this code
please cite this paper:
@ARTICLE{9624977,
  author={Qu, Jiahui and Hou, Shaoxiong and Dong, Wenqian and Li, Yunsong and Xie, Weiying},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={A Multilevel Encoder–Decoder Attention Network for Change Detection in Hyperspectral Images},
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2021.3130122}}
"""


import torch
import torch.nn as nn


## ablation experimtent
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, stride=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttentionBasicBlock(nn.Module):
    def __init__(self, gc):
        super(AttentionBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(gc, gc, 3, 1, 1)
        self.relu = nn.PReLU()
        self.Channelattention = ChannelAttention(gc)
        self.Spatitalattention = SpatialAttention(7)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = x * self.Channelattention(x)
        out = out * self.Spatitalattention(out)
        out = x + out
        out = self.relu(self.conv2(out))
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)

    def forward(self, x):
        r_out, (h, c) = self.rnn(x, None)

        return r_out[:, -1, :]

class ML_EDAN(nn.Module):
    def __init__(self, in_channel):
        super(ML_EDAN, self).__init__()
        ## AE model
        self.cnn1 = AE(in_channel)
        self.cnn2 = AE(in_channel)
        ## LSTM
        self.lstm1 = RNN(6400, 512)
        self.lstm2 = RNN(2304, 256)
        self.lstm3 = RNN(512, 128)
        self.lstm4 = RNN(4608, 512)
        self.lstm5 = RNN(12800, 1024)
        ## FC
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear = nn.Linear(416, 64)
        self.linear1_1 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, T1, T2):
        T1_out3, T1_out4, T1_out5, T1_out6 = self.cnn1(T1)
        T2_out3, T2_out4, T2_out5, T2_out6 = self.cnn2(T2)
        out_3 = torch.cat(
            [T1_out3.view(T1_out3.size(0), -1).unsqueeze(1), T2_out3.view(T2_out3.size(0), -1).unsqueeze(1)], dim=1)
        out_3 = self.lstm3(out_3)
        out_4 = torch.cat(
            [T1_out4.view(T1_out4.size(0), -1).unsqueeze(1), T2_out4.view(T2_out4.size(0), -1).unsqueeze(1)], dim=1)
        out_4 = self.lstm4(out_4)
        out_5 = torch.cat(
            [T1_out5.view(T1_out5.size(0), -1).unsqueeze(1), T2_out5.view(T2_out5.size(0), -1).unsqueeze(1)], dim=1)
        out_5 = self.lstm5(out_5)
        out_15 = self.linear1(out_5)
        out_24 = self.linear2(out_4)
        out_33 = self.linear3(out_3)
        out = torch.cat([out_33, out_24, out_15], dim=1)
        out = self.linear1_1(self.relu(self.linear(out)))
        return out, T1_out6, T2_out6


### Autoencoder
class AE(nn.Module):
    def __init__(self, in_channel):
        super(AE, self).__init__()
        ## encoder
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        ## decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=in_channel, kernel_size=3, stride=1, padding=1)
        ## AttentionBlock
        self.attentionblock1 = AttentionBasicBlock(256)
        self.attentionblock2 = AttentionBasicBlock(256)
        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(size = (3, 3), mode = 'nearest')
        self.Conv_attention1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1)
        self.trans1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.upsample2 = nn.Upsample(size=(5, 5), mode = 'nearest')
        self.Conv_attention2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.trans2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.deconv1(out3))
        ## CIGA Attention
        out3_4 = self.sigmoid(self.Conv_attention1(torch.cat([out2, self.upsample1(out3)], dim=1))) * out2
        out4 = self.relu(self.trans1(torch.cat([out3_4, out4], dim=1)))
        ## CIGA Attention
        out3_5 = self.sigmoid(self.Conv_attention2(torch.cat([out1, self.upsample2(out4)], dim=1))) * out1
        out5 = self.relu(self.deconv2(out4))
        out5 = self.relu(self.trans2(torch.cat([out3_5, out5], dim=1)))
        out6 = self.deconv3(out5)
        out_24 = torch.cat([out2, out4], dim=1)
        out_15 = torch.cat([out1, out5], dim=1)
        ### return three middle feature maps and the reconstructed feature map
        return out3, out_24, out_15, out6


if __name__ == "__main__":
    ### model test
    T1 = torch.randn(1, 154, 5, 5)
    T2 = torch.randn(1, 154, 5, 5)
    ## china:154 River:198 BayArea:224
    model = ML_EDAN(in_channel=154)
    out, T1, T2 = model(T1, T2)
    print("out.shape:", out.shape, "T1.shape:", T1.shape, "T2.shape:", T2.shape)
