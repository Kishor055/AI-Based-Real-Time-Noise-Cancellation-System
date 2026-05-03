import torch
import torch.nn as nn

class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoiseNet()

def denoise_model(signal):
    with torch.no_grad():
        x = torch.tensor(signal[:1024]).float()
        out = model(x)
    return out.numpy()
