'''
Now we have the data all we haveto do i to turn our datat into pyotch dataset 
and pyhtorch dataloader objects.
'''
import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
num_worker = os.cpu_count() if os.cpu_count() else 0

def create_dataloaders(train_dir:str,
                       test_dir:str,
                       transform: transforms.Compose,
                       batch_size: int):
    """
    create a training and testing dataloader form the training and testing directories
    args:
    train_dir: str - path to training directory
    test_dir: str - path to testing directory
    transform: transforms.Compose - transformations to apply to the data
    batch_size: int - batch size for the dataloader
    num_workers: int - number of workers to use for the dataloader
    """
    # create training and testing datasets
    train_data = datasets.ImageFolder(root =train_dir,transform=transform)
    test_data = datasets.ImageFolder(root =test_dir,transform=transform)
    # creating the training and testing data laoders
    train_dataloader =DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True
    )
    return train_dataloader , test_dataloader

if __name__ == "__main__":
    # setup paths to training and testing directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"
    
    # setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=32,
        num_workers=num_worker
    )
    
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of testing batches: {len(test_dataloader)}") 