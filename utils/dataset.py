import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_mnist(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train

def select_from_dataset(dataset, per_class_size, labels):
    indices_by_label = [[] for _ in range(10)]

    for i in range(len(dataset)):
        current_class = dataset[i][1]
        indices_by_label[current_class].append(i)
    indices_of_desired_labels = [indices_by_label[i] for i in labels]

    return Subset(dataset, [item for sublist in indices_of_desired_labels for item in sublist[:per_class_size]])

def load_fmnist(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.FashionMNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train

def load_celeba(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    celeba_train = torchvision.datasets.CelebA(root=file_location, target_type="identity", download=False, transform=transform)
    return celeba_train

def select_from_celeba(dataset, size):
    return Subset(dataset, range(size))
