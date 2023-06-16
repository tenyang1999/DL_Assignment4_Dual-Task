
import numpy as np  
import os
import torch
import json
from PIL import Image
import pickle

with open('ADE20k_classes.pkl', 'rb') as f:
    classes = pickle.load(f)

class ADEDataset(torch.utils.data.Dataset):

    def __init__(self, root,  filename, transforms):
        self.root = root
        self.transforms = transforms
        self.image_list = self.read_file(os.path.join(root, filename))
        self.len = len(self.image_list)

    def __getitem__(self, idx):
        # load images and masks
        image_name= self.image_list[idx]
        img_path = os.path.join(self.root, "imgs", f"ADE_val_{image_name}.jpg")
        img = Image.open(img_path).convert("RGB")
        json_path = os.path.join(self.root, "jsons", f"ADE_val_{image_name}.json")
        mask_path = os.path.join(self.root, "instance_mask_backup")
        
        with open(json_path) as f:
            img_data = json.load(f)

        label = []
        masks = []
        boxes = []
        for obj in img_data['annotation']['object']:
            id = obj['id']
            obj_name = obj['name'].split(",")
            for single_name in obj_name:    
                single_name = single_name.strip() 
                if  single_name in classes.keys():
                    xmin = min(obj['polygon']['x'])
                    xmax = max(obj['polygon']['x'])
                    ymin = min(obj['polygon']['y'])
                    ymax = max(obj['polygon']['y'])
                    if xmin == xmax or ymin == ymax:
                        break
                        
                    boxes.append([xmin, ymin, xmax, ymax])
                    
                    label.append(classes[single_name])

                    instance_path = os.path.join(mask_path, obj["instance_mask"])
                    mask = np.array(Image.open(instance_path))
                    masks.append(mask)
                    break

        boxes = np.array(boxes)
        # convert everything into a torch.Tensor
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(np.array(label), dtype=torch.int64) - 1
        target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        target["area"] =torch.as_tensor(area, dtype=torch.uint8)
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_list)
    
    def read_file(self,filename):
        image_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                img = line.rstrip().split(' ')[0]
                image_list.append(img)
        return image_list
    
    


