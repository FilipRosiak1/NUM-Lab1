import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import torch

class EmojiDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = int(img_name.split('-')[0])
        image = Image.open(os.path.join(self.data_dir, img_name)).convert("RGBA") # load image as RGBA

        # convert to RGB with white background
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(white_bg.convert("RGBA"), image).convert("RGB") 

        if self.transform:
            image = self.transform(image)
        return image, label

class EmojiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # resize image to 128x128 and do random horizontal flip
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                              std=[0.5, 0.5, 0.5])
        ])
        
    def setup(self, stage=None):
        full_dataset = EmojiDataset(self.data_dir, transform=self.transform)
        
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_split)
        val_size = int(dataset_size * self.val_split)
        test_size = dataset_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Rozmiar całego zbioru: {dataset_size}")
        print(f"Rozmiar zbioru treningowego: {train_size} ({self.train_split*100}%)")
        print(f"Rozmiar zbioru walidacyjnego: {val_size} ({self.val_split*100}%)")
        print(f"Rozmiar zbioru testowego: {test_size} ({self.test_split*100}%)")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

# for testing
if __name__ == "__main__":
    data_module = EmojiDataModule(
        data_dir="dataset",
        batch_size=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    data_module.setup() 