from torchvision import datasets, transforms

def get_dataset(dataset_path='../data'):
    tr_dataset = datasets.MNIST(dataset_path, train=True, download=False, transform=transforms.ToTensor())

    te_dataset = datasets.MNIST(dataset_path, train=False, download=False, transform=transforms.ToTensor())
    
    return tr_dataset, te_dataset