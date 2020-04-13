import torch
from torch import nn, optim

from network import VAE, loss_function
from mnist_data import get_dataset
import hyperparams as hp
from tqdm import tqdm
from util import logging

def main():
    tr_dataset, te_dataset = get_dataset(hp.dataset_path)
    
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    
    for epoch in range(1, hp.epochs+1):
        model.train()
        dataloader = torch.utils.data.DataLoader(tr_dataset, 
                                                  batch_size=hp.batch_size, 
                                                  shuffle=True,
                                                  num_workers=8,
                                                  pin_memory=True)

        tr_loss = 0
        pbar = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar, 1):
            pbar.set_description(f'Train epoch {epoch : 3d}')
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()

            tr_loss += loss.item()
            optimizer.step()        

            pbar.set_postfix({'loss' : f'{tr_loss / (batch_idx * hp.batch_size) : .3f}'})

        logging(f'Epoch {epoch} : train loss = {tr_loss/len(dataloader.dataset) : .3f}')

        if epoch%hp.log_interval == 0:
            model.eval()
            dataloader = torch.utils.data.DataLoader(te_dataset, 
                                                      batch_size=hp.batch_size, 
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=True)

            te_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader, 1):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)   
                te_loss += loss.item()

            logging(f'Epoch {epoch} : test loss = {te_loss/len(dataloader.dataset) : .3f}')

if __name__ == '__main__':
    use_cuda = not hp.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    torch.manual_seed(hp.seed)
    
    main()