import zipfile
import os
import sys
import io
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROIS_ZIP, FEATURES_RESNET_DIR

OUTPUT_FEATURES = FEATURES_RESNET_DIR

os.makedirs(OUTPUT_FEATURES, exist_ok=True)

# --- ResNet50 Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Usando dispositivo: {DEVICE}")

class ResNet50FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.resnet50(pretrained=True)
            
        self.features = torch.nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3
        )
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.features(x)
        # Interpolate spatial dimension from 65x65 to 74x74 to align with DinoV2 features
        x = F.interpolate(x, size=(74, 74), mode='bilinear', align_corners=False)
        return x

# Custom dataset to read directly from the ZIP in parallel
class ImageZipDataset(Dataset):
    def __init__(self, zip_path, transform):
        self.zip_path = zip_path
        self.transform = transform
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            self.tifs = sorted([f for f in z.namelist() if f.endswith('.tif')])
            
    def __len__(self):
        return len(self.tifs)
        
    def __getitem__(self, idx):
        tif_path = self.tifs[idx]
        image_id = os.path.basename(tif_path).replace(".tif", "")
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            with z.open(tif_path) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        tensor = self.transform(img)
        return image_id, tensor

TRANSFORM = T.Compose([
    T.Resize((1036, 1036)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_all():
    print("Starting parallel ResNet-50 feature extraction...")
    
    # Instantiate extractor and move it to the device
    model = ResNet50FeatureExtractor().to(DEVICE)
    model.eval()
    
    dataset = ImageZipDataset(ROIS_ZIP, TRANSFORM)
    # Using num_workers=4 to parallelize loading and transforms on CPU
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    
    with torch.no_grad():
        for ids, tensors in tqdm(dataloader, desc="Extraindo features ResNet-50"):
            # Filter out images whose features already exist, to speed up a resumed run
            existing_mask = [os.path.exists(os.path.join(OUTPUT_FEATURES, f"{image_id}.pt")) for image_id in ids]
            
            # If every item in the batch already exists, skip the forward pass
            if all(existing_mask):
                continue
                
            tensors = tensors.to(DEVICE)
            features = model(tensors)
            
            for i, image_id in enumerate(ids):
                feature_path = os.path.join(OUTPUT_FEATURES, f"{image_id}.pt")
                if not os.path.exists(feature_path):
                    torch.save(features[i:i+1].cpu(), feature_path)

if __name__ == "__main__":
    process_all()
    print("ResNet-50 extraction complete.")
