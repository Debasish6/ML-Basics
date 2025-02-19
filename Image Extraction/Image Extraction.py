import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

root = r"C:/Users/eDominer/Python Project/Products/"
images = os.listdir(root)

model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

# Check if a GPU is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class to handle image loading and transformation
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path
        except Exception as e :
            return None,img_path


# Function to extract features from the model's avgpool layer
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

# Function to process images and extract feature vectors
def extract_features(image_paths, batch_size=32):
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_names = []
    all_vecs = []

    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)

            _ = model(images)

            vecs = activation["avgpool"].cpu().numpy()

            all_vecs.append(vecs)
            all_names.extend(paths)

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size} images...")

    all_vecs = np.vstack(all_vecs)
    return all_names, all_vecs

image_paths = [os.path.join(root, img) for img in images]
all_names, all_vecs = extract_features(image_paths)

np.save(f"C:/Users/eDominer/Python Project/Image Extraction/all_vecs1.npy", all_vecs)
np.save(f"C:/Users/eDominer/Python Project/Image Extraction/all_names1.npy", all_names)

print("Feature extraction completed and results saved!")
