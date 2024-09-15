import torch
import torch.nn as nn


from .utils1 import Conv1d, MaxPool1d
from position_encoding import positional_encoding
from mamba_ssm import Mamba
class DeepSleepNetFeature_from_sleepyco(nn.Module):
    def __init__(self):
        super(DeepSleepNetFeature_from_sleepyco, self).__init__()

        self.chn = 64

        
        self.path1 = nn.Sequential(Conv1d(2, self.chn, 50, 6, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(8, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME')
                                   )
        self.path2 = nn.Sequential(Conv1d(2, self.chn, 400, 50, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(2, padding='SAME'))

        self.compress = nn.Conv1d(self.chn*4, 128, 1, 1, 0)
        
        self.conv_c5 = nn.Conv1d(128, 128, 1, 1, 0)
        self.fc = nn.Linear(128 * 16, 5)
        c=128
        t_d=16
        t=16
        c_d=128
        n_heads=4
        dropout=0.1
        self.mamba=Mamba(d_model=2,d_state=32,d_conv=4,expand=2)
        
    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.mamba(x)
        x=x.permute(0,2,1)
        
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        x2 = torch.nn.functional.interpolate(x2, x1.size(2))
        c5 = self.compress(torch.cat([x1, x2], dim=1))
        out = self.conv_c5(c5)
        out=self.fc(out.reshape(out.shape[0], -1))

        return out
if __name__ == '__main__':
    from torchinfo import summary
    model=DeepSleepNetFeature_from_sleepyco()
    model.to('cuda')
    batch_size=64
    model(torch.randn(batch_size, 2, 3000).to('cuda'))
    