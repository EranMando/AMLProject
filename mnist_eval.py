import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import config
from temporal_ensembling import train
from temporal_ensembling_improved import trainimp
from utils import GaussianNoise, savetime, save_exp
from utils import calc_metrics, prepare_mnist, weight_schedule,prepare_kmnist,prepare_fashion_mnist,prepare_emnist


class CNN(nn.Module):
    
    def __init__(self, batch_size, std, p=0.5, fm1=16, fm2=32):
        super(CNN, self).__init__()
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.gn    = GaussianNoise(batch_size, std=self.std)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding='valid'))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding='valid'))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10)
    
    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


# metrics
accs         = []
accs_best    = []
losses       = []
sup_losses   = []
unsup_losses = []
idxs         = []


ts = savetime()
cfg = vars(config)

for i in range(cfg['n_exp']):
    model = CNN(cfg['batch_size'], cfg['std'])
    seed = cfg['seeds'][i]
    acc, acc_best, l, sl, usl, indices = train(model, seed,prepare_mnist, **cfg)
    accs.append(acc)
    accs_best.append(acc_best)
    losses.append(l)
    sup_losses.append(sl)
    unsup_losses.append(usl)
    idxs.append(indices)

print('saving experiment'    )

save_exp(ts, losses, sup_losses, unsup_losses,
         accs, accs_best, idxs, **cfg)

