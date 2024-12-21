from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms
import numpy as np
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

def get_dataloader(dir, tranform=transform,batch_size=16, num_workers=2):
   data = ImageDataset(dir, tranform)
   loader = DataLoader(
    data,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True
   )
   return loader

def cosine_similarity(vector1, vector2):
  a = np.dot(vector1, vector2)
  b = np.linalg.norm(vector1)
  c = np.linalg.norm(vector2)
  if b == 0 or c == 0:
    return 1.0
  return 1-(a / (c * b))