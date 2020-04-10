import torch
import torch.optim as optim
import torch.nn.functional as F

from network import Net
from mnist_data import get_dataset

import hyperparams as hp
from tqdm import tqdm

def main():
    tr_dataset, te_dataset = get_dataset(hp.dataset_path)
    
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hp.lr)
    
    for epoch in range(1, hp.epochs+1):
        model.train()
        dataloader = torch.utils.data.DataLoader(tr_dataset, 
                                                  batch_size=hp.tr_batch_size, 
                                                  shuffle=True,
                                                  num_workers=8,
                                                  pin_memory=True)

        tr_loss = 0
        correct = 0
        pbar = tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar, 1):
            pbar.set_description(f'Train epoch {epoch : 3d}')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            tr_loss += loss
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

            pbar.set_postfix({'loss' : f'{tr_loss.item() / (batch_idx * hp.tr_batch_size) : .3f}',
                              'correct' : f'{correct.item() / (batch_idx * hp.tr_batch_size) : .3f}'})

        if epoch%hp.log_interval == 0:
            model.eval()
            dataloader = torch.utils.data.DataLoader(te_dataset, 
                                                      batch_size=hp.te_batch_size, 
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=True)

            te_loss = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(dataloader, 1):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)    
                te_loss += loss

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()


            print(f'Epoch {epoch} : test loss = {te_loss/len(dataloader.dataset) : .3f}, test correct = {correct/len(dataloader.dataset) : .3f}')

if __name__ == '__main__':
    use_cuda = not hp.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    torch.manual_seed(hp.seed)
    
    main()