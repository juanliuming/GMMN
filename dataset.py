import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

train_label_file = 'Disease Grading/a. IDRiD_Disease Grading_Training Labels.xlsx'  
val_label_file = 'Disease Grading/b. IDRiD_Disease Grading_Testing Labels.xlsx' 
train_image_folder = 'Disease Grading/1. Original Images/a. Training Set'  
val_image_folder = 'Disease Grading/1. Original Images/b. Testing Set' 

train_df = pd.read_excel(train_label_file)
val_df = pd.read_excel(val_label_file)

train_image_names = train_df['image_name'].values
train_labels = train_df['label'].values
val_image_names = val_df['image_name'].values
val_labels = val_df['label'].values

train_data = list(zip(train_image_names, train_labels))
val_data = list(zip(val_image_names, val_labels))

class CustomImageDataset(Dataset):
    def __init__(self, data, image_folder, transform=None):
        self.data = data
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, label

transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def dataloader(batch_size):
    train_dataset = CustomImageDataset(train_data, train_image_folder, transform=transform_train)
    val_dataset = CustomImageDataset(val_data, val_image_folder, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,prefetch_factor=8,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12,prefetch_factor=8,pin_memory=True,persistent_workers=True)
    return train_loader, val_loader

if __name__ == '__main__':
    batch_size = 2
    train_loader, val_loader = dataloader(batch_size)
    for images, labels in train_loader:
        print(images[0,1,:,50])
        break
