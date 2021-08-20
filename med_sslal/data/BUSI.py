import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class BUSIDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted([f for f in os.listdir(os.path.join(root, "images")) if not f.startswith('.')]))
        self.masks = self.imgs  

        lesion_class_dict = {'normal': 0, 'benign': 1, 'malignant': 2}   
        lesion_classes = [img_name.split('_')[0] for img_name in self.imgs]
        self.class_distribution = np.unique(lesion_classes, return_counts=True)
        self.class_labels = [lesion_class_dict[c] for c in lesion_classes]

    def get_class_distribution(self):
        return self.class_distribution

    def __getitem__(self, idx):
        # Read image and its corresponding mask
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask_ground_truth", self.imgs[idx])

        img = cv2.imread(img_path).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        # Normalize and scale the image to [0, 1]
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        img = np.clip(img, -1.0, 1.0)
        img = (img + 1.0) / 2.0   

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # expected image dimension: a 3d tensor [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        # First id is the background, remove it
        # The following steps are optional, but work for both Faster & Mask R-CNN
        obj_ids = np.unique(mask)[1:] 
        num_objs = len(obj_ids)

        # Read class labels
        labels = [self.class_labels[idx]]

        # For negative samples (background only)
        if len(obj_ids) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            area = np.zeros(0, dtype=np.float32)
        # For positive samples
        else: 
            masks = mask == obj_ids[:, None, None]
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            area = (boxes[0][3] - boxes[0][1]) * (boxes[0][2] - boxes[0][0])

        # convert everything into a torch.Tensor
        img = torch.as_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.tensor([area])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.imgs)