import os
import torch
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

IMG_PATH = '/scratch/diogo.alves/datasets/brset/physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos'
CSV_DIR = '/home/gabrieljp/BRSET/few-shot-BRSET/data'
CLASS_COLUMNS = ['hemorrhage', 'vascular_occlusion','diabetic_retinopathy',
                 'macular_edema', 'scar', 'nevus', 'amd', 
                 'hypertensive_retinopathy', 'drusens', 
                 'myopic_fundus', 'increased_cup_disc']
SPLITS = ['test', '10-shot', '20-shot', '40-shot', 'all']

class FewShotBRSET(Dataset):
    def __init__(self, img_dir=IMG_PATH, csv_dir=CSV_DIR, transform=None, tasks = CLASS_COLUMNS, split = '40-shot' ):
        
        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.transform = transform
        self.tasks = tasks
        self.split = split

        if isinstance(self.tasks, str):
            self.tasks = [self.tasks]

        assert set(self.tasks) <= set(CLASS_COLUMNS)
        assert self.split in SPLITS

        self.csv = pd.concat([pd.read_csv(f"{self.csv_dir}/{task}.csv") for task in self.tasks])

        if self.split == 'test' or self.split == '10-shot':
            self.csv = self.csv[self.csv.split == self.split]
        elif self.split == '20-shot':
            self.csv = self.csv[(self.csv.split == self.split) | (self.csv.split == '10-shot')]
        elif self.split == '40-shot':
            self.csv = self.csv[(self.csv.split == self.split) | (self.csv.split == '20-shot')| (self.csv.split == '10-shot')]

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                f"{self.csv.iloc[idx]['image_id']}.jpg")
        image = read_image(img_name)
        label = int(1 in set(self.csv.iloc[idx][CLASS_COLUMNS]))

        if self.transform:
            image = self.transform(image)

        return image, label

ds = FewShotBRSET(split='40-shot', tasks='hemorrhage')
print(ds[0][1])
print(ds[-1][1])

        