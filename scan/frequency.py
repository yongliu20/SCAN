import torch
import torch.nn as nn
import torch.nn.functional as F

class LFM(nn.Module):
    def __init__(self, num_channels):
        super(LFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()


        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)       #1, 1, H, W


    def forward(self, x, sigma):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)


        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w, sigma=sigma)
        y = y * (1 - high_filter)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class CA(nn.Module):
    def __init__(self, input_dim, num):
        super(CA, self).__init__()
        self.num = num
        self.multiattn = nn.ModuleList()
        self.ln = nn.ModuleList()
        for i in range(num):
            self.multiattn.append(nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True))
            if i != num - 1:
                self.ln.append(nn.LayerNorm(input_dim))
    
    def forward(self, tgt, memory):
        for i in range(self.num):
            tgt = tgt + self.multiattn[i](tgt, memory, memory)[0]
            if i != self.num - 1:
                tgt = self.ln[i](tgt)
        return tgt