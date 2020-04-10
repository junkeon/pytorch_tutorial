from torchvision import datasets, transforms

def get_dataset(dataset_path='../data'):
    tr_dataset = datasets.MNIST(dataset_path, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    te_dataset = datasets.MNIST(dataset_path, train=False, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    
    return tr_dataset, te_dataset