import os
from PIL import Image
from torch.utils.data import Dataset
import random

class ImagenetteDataset(Dataset):
    def __init__(self, txt_path, base_img_dir):
        """
        :param txt_path: path to val.txt
        :param base_img_dir: './DATA/imagenette2/imagenette2-160/val'
        """
        self.base_img_dir = base_img_dir
        
        with open(txt_path, 'r', encoding='utf-8') as file:
            self.lines = file.read().splitlines()
            
        self.id2label = {
            "n01440764": "tench", 
            "n02102040": "English springer", 
            "n02979186": "cassette player", 
            "n03000684": "chain saw", 
            "n03028079": "church", 
            "n03394916": "French horn", 
            "n03417042": "garbage truck", 
            "n03425413": "gas pump", 
            "n03445777": "golf ball", 
            "n03888257": "parachute"
        }

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        rel_path = self.lines[idx]
        img_path = os.path.join(self.base_img_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        class_id = rel_path.split('/')[-2] 
        label = self.id2label[class_id]
        return image, label, img_path

if __name__ == '__main__':
    txt_file_path = './DATA/imagenette2/val.txt'
    image_root_dir = './DATA/imagenette2' 
    val_dataset = ImagenetteDataset(txt_file_path, image_root_dir)
    print(f"Data Num: {len(val_dataset)}")
    print("\n-----Randomly choose one image-----")
    img, lbl, img_path = random.choice(val_dataset)
    print(f"Image Path: {img_path}")
    print(f"Label: {lbl}")
    print(f"Image Size: {img.size}")