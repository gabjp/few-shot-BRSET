import os
import torch
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random 
from sklearn.model_selection import train_test_split

class FewShotBRSET(Dataset):
    def __init__(self, img_ids, labels, transforms=None):
        self.img_ids = img_ids
        self.transforms = transforms
        self.labels = labels
        self.img_dir = '/home/gabrieljp/BRSET/few-shot-BRSET/data_2/imgs'
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                f"{self.img_ids[idx]}.jpg")
        image = read_image(img_name) / 255
        label = self.labels[idx].float()

        if self.transforms:
            image = self.transforms(image)

        return image, label


class BRSETManager():
    def __init__(self, training_classes, test_classes, n_shots, n_ways, mean, std, augment = None, batch_size = 25):
        self.training_classes = training_classes
        self.test_classes = test_classes
        self.n_shots = n_shots
        self.n_ways = n_ways
        self.augment = augment
        self.data = pd.read_csv('/home/gabrieljp/BRSET/few-shot-BRSET/data_2/clean.csv')
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
    
    def get_train_task(self):
        classes = random.sample(self.training_classes, self.n_ways)
        train_ids = []
        train_labels = torch.empty((0, self.n_ways), dtype=torch.float32)

        for clss in classes:
            objs = self.data[self.data[clss] == 1].sample(n=self.n_shots)
            train_ids += list(objs["image_id"])
            train_labels = torch.cat((train_labels,torch.tensor(objs[classes].to_numpy())), axis = 0) 

        if self.augment == None:
            dataset = FewShotBRSET(train_ids, train_labels, transforms = transforms.Normalize(self.mean, self.std))
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else: 
            raise NotImplementedError("augmentation not implemented yet")

        return train_loader
    
    def get_eval_task(self):
        classes = random.sample(self.test_classes, self.n_ways)
        train_ids = []
        train_labels = torch.empty((0, self.n_ways), dtype=torch.float32)
        test_ids = []
        test_labels = []

        for clss in classes:
            objs = self.data[self.data[clss] == 1].sample(n=self.n_shots+1)

            train_ids += list(objs["image_id"])[:-1]
            train_labels = torch.cat((train_labels, torch.tensor(objs[classes].to_numpy()[:-1])), axis = 0)
            
            test_ids.append(list(objs["image_id"])[-1])
            test_labels.append(torch.tensor(objs[classes].to_numpy()[-1]))
        
        test_labels = torch.stack(test_labels)
            
        if self.augment == None:
            dataset = FewShotBRSET(train_ids, train_labels, transforms = transforms.Normalize(self.mean, self.std))
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else: 
            raise NotImplementedError("augmentation not implemented yet")
        
        dataset = FewShotBRSET(test_ids, test_labels, transforms = transforms.Normalize(self.mean, self.std))
        test_loader = DataLoader(dataset, batch_size=self.n_ways, shuffle=True)

        return train_loader, test_loader, classes

    def get_ss_split(self, val_ratio=0.1):
        ds = self.data[self.data[self.training_classes].sum(axis = 1) == 1]
        img_ids = list(ds['image_id'])
        img_labels = torch.tensor(ds[self.training_classes].to_numpy())
        ids_train, ids_test, labels_train, labels_test = train_test_split(img_ids, img_labels, test_size = val_ratio,
                                                                          random_state=42, stratify = img_labels)
        train_set = FewShotBRSET(ids_train, labels_train, transforms = transforms.Normalize(self.mean, self.std))
        test_set = FewShotBRSET(ids_test, labels_test, transforms = transforms.Normalize(self.mean, self.std))

        train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, self.batch_size)

        return train_loader, test_loader


if __name__ == "__main__":
    training_classes = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
    test_classes = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']
    n_shots = 5
    n_ways = 2
    manager = BRSETManager(training_classes, test_classes, n_shots, n_ways, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), augment = None, batch_size = 10)

    train_task = manager.get_train_task()

    for img, label in train_task:
        print(img.size())
        print(label.size())

    print()

    train_task, test_task = manager.get_eval_task()

    for img, label in train_task:
        print(img.size())
        print(label)

    print()

    for img, label in test_task:
        print(img.size())
        print(label.size())




        

        



