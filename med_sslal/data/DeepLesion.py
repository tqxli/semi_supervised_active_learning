import torch
from torch.utils.data import Dataset

import numpy as np
import cv2
import os
import pandas as pd
import re


"""
The DeepLesion images are stored in unsigned 16 bit. 
One should subtract 32768 from the pixel intensity to obtain the original Hounsfield unit (HU) values.
"""
read_hu = lambda x: cv2.imread(x).astype(np.float32) - 32768

class DeepLesionDataset(Dataset):
    def __init__ (self, root="/content/drive/MyDrive/data/DeepLesion",
                        transforms=None, 
                        dataset_type="train", 
                        lesion_type="all_types"):

        self.img_path = os.path.join(root, 'Images_png') 
        self.csv_path = os.path.join(root, 'annotations/DL_info.csv')

        dataset_type_dict = {"train": 1, "val": 2, "test": 3, "non-specified": 0}
        lesion_type_dict = {"bone": 1, "abdomen": 2, "mediastinum": 3, "liver": 4, "lung": 5, "kidney": 6, "soft_tissue": 7, "pelvis": 8, "all_types": 0}

        self.df = pd.read_csv(self.csv_path)
        self.dataset_type = dataset_type_dict[dataset_type]
        if self.dataset_type != 0:
            self.df = self.df[self.df['Train_Val_Test'] == self.dataset_type]

        self.lesion_type = lesion_type_dict[lesion_type]
        if self.lesion_type != 0:
            self.df = self.df[self.df['Coarse_lesion_type'] == self.lesion_type]
        
        self.df['img_path'] = self.df.apply(lambda c_row: os.path.join(self.img_path, 
                                                                  '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row),
                                                                  '{Key_slice_index:03d}.png'.format(**c_row)), 1)
        self.df = self.df.reset_index()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def get_num_classes(self):
        return len(self.df['Coarse_lesion_type'].unique())

    def __getitem__(self, idx):
        # Read image path from the csv annotation file
        img_path = self.df.iloc[idx]['img_path']    
        img = read_hu(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale to [0, 1] and normalize
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        img = np.clip(img, -1.0, 1.0)
        img = (img + 1.0) / 2.0

        # Read bounding box coordinates (1-3 bboxes per image)
        # [x1, y1, x2, y2]
        boxes = []
        new_df = self.df[self.df['File_name']==self.df.iloc[idx]['File_name']]
        num_objs = len(new_df.index)
        for i in range(num_objs):
            coordinates_str = (re.split(',', new_df.iloc[i]['Bounding_boxes']))
            coordinates = [float(x) for x in coordinates_str]
            boxes.append(coordinates)
        boxes = np.asarray(boxes)

        labels = np.ones((num_objs, ), dtype=np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img, boxes = transformed['image'], transformed['bboxes']

        # Convert everything into torch.Tensors
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((1, 1), dtype=torch.int64)
            area = torch.tensor([0], dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else: 
            labels = torch.tensor(labels)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Targets
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target