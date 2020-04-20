import torch
from torch import nn, optim

from network import Sequence
import hyperparams as hp
from tqdm import tqdm
import numpy as np

from generate_sine_wave import generate_sine

import matplotlib.pyplot as plt

def main(file_name):
    data = torch.load(file_name)
    
    tr_inputs = torch.from_numpy(data[3:, :-1]).to(device)
    tr_targets = torch.from_numpy(data[3:, 1:]).to(device)

    te_inputs = torch.from_numpy(data[:3, :-1]).to(device)
    te_targets = torch.from_numpy(data[:3, 1:]).to(device)
    
    model = Sequence(device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    
    plt.figure(figsize = (10, 30))

    for epoch in range(1, hp.epochs+1):
        model.train()
        print(f'Epoch {epoch} : ', end='')
        optimizer.zero_grad()
        tr_output = model(tr_inputs)
        loss = criterion(tr_output, tr_targets)
        loss.backward()
        optimizer.step()
        print(f'tr_loss = {loss.item():.3f}', end='')

        if epoch % hp.log_interval == 0:
            model.eval()
            with torch.no_grad():
                future = 1000
                pred = model(te_inputs, future=future)
                loss = criterion(pred[:, :-future], te_targets)
                print(f', te_loss = {loss.item():.3f}')

                y = pred.detach().cpu().numpy()

            i = epoch // hp.log_interval

            plt.subplot(hp.epochs//hp.log_interval, 1, i)
            plt.title(f'Epoch {epoch}')

            for i, color in enumerate(['r', 'g', 'b']):
                plt.plot(np.arange(te_inputs.size(1)), y[i][:te_inputs.size(1)], color, linewidth = 2.0)
                plt.plot(np.arange(te_inputs.size(1), te_inputs.size(1) + future), y[i][te_inputs.size(1):], color + ':', linewidth = 2.0)
        print()
    plt.savefig('sine_prediction.png')

if __name__ == '__main__':
    use_cuda = not hp.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    torch.manual_seed(hp.seed)
    
    file_name = hp.f_name
    generate_sine(file_name)
    
    main(file_name)