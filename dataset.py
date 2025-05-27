import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 读取标签文件
train_label_file = 'Disease Grading/a. IDRiD_Disease Grading_Training Labels.xlsx'  # 训练集标签文件路径
val_label_file = 'Disease Grading/b. IDRiD_Disease Grading_Testing Labels.xlsx'  # 验证集标签文件路径
train_image_folder = 'Disease Grading/1. Original Images/a. Training Set'  # 训练集图像文件夹路径
val_image_folder = 'Disease Grading/1. Original Images/b. Testing Set'  # 验证集图像文件夹路径

# 读取Excel中的数据
train_df = pd.read_excel(train_label_file)
val_df = pd.read_excel(val_label_file)

# 假设Excel中有两列：image_name 和 label
train_image_names = train_df['image_name'].values
train_labels = train_df['label'].values
val_image_names = val_df['image_name'].values
val_labels = val_df['label'].values

# 将图像名称和标签配对
train_data = list(zip(train_image_names, train_labels))
val_data = list(zip(val_image_names, val_labels))

# 自定义Dataset类
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
        image = Image.open(image_path).convert('RGB')  # 假设图像为RGB格式
        if self.transform:
            image = self.transform(image)
        return image, label

# 图像转换，例如：调整大小并转为tensor
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

# 创建DataLoader
def dataloader(batch_size):
    # 创建训练集和验证集的Dataset
    train_dataset = CustomImageDataset(train_data, train_image_folder, transform=transform_train)
    val_dataset = CustomImageDataset(val_data, val_image_folder, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,prefetch_factor=8,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12,prefetch_factor=8,pin_memory=True,persistent_workers=True)
    return train_loader, val_loader

# 验证读取是否正确
if __name__ == '__main__':
    batch_size = 2
    train_loader, val_loader = dataloader(batch_size)
    for images, labels in train_loader:
        print(images[0,1,:,50])
        break