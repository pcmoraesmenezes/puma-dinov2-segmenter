import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np


class PUMAFeatureDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.feature_dir = os.path.join(base_path, "features")
        self.mask_tissue_dir = os.path.join(base_path, "masks/tissue")
        self.mask_nuclei_dir = os.path.join(base_path, "masks/nuclei")
        
        # Lista de IDs baseada nos arquivos de features
        self.sample_ids = sorted([f.replace(".pt", "") for f in os.listdir(self.feature_dir) if f.endswith(".pt")])
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 1. Carregar Features (C, H, W) -> (384, 74, 74)
        feature_path = os.path.join(self.feature_dir, f"{sample_id}.pt")
        features = torch.load(feature_path).squeeze(0) # Remove dim de batch se houver
        
        # 2. Carregar Máscaras
        tissue_mask_path = os.path.join(self.mask_tissue_dir, f"{sample_id}.png")
        nuclei_mask_path = os.path.join(self.mask_nuclei_dir, f"{sample_id}.png")
        
        # Máscaras são salvas em (1024, 1024)
        tissue_mask = cv2.imread(tissue_mask_path, cv2.IMREAD_UNCHANGED)
        nuclei_mask = cv2.imread(nuclei_mask_path, cv2.IMREAD_UNCHANGED)
        
        # Converter para Tensores Long (Labels de classificação)
        tissue_mask = torch.from_numpy(tissue_mask).long()
        nuclei_mask = torch.from_numpy(nuclei_mask).long()
        
        return {
            "id": sample_id,
            "features": features,
            "mask_tissue": tissue_mask,
            "mask_nuclei": nuclei_mask
        }
        

if __name__ == "__main__":
    # Teste Rápido
    BASE_PATH = "/home/paulo/Área de Trabalho/repo-pessoal/puma-dinov2-segmenter/puma"
    dataset = PUMAFeatureDataset(BASE_PATH)
    print(f"📦 Dataset inicializado com {len(dataset)} amostras.")
    
    sample = dataset[0]
    print(f"Sample ID: {sample['id']}")
    print(f"Features Shape: {sample['features'].shape}")
    print(f"Tissue Mask Shape: {sample['mask_tissue'].shape}")
    print(f"Nuclei Mask Shape: {sample['mask_nuclei'].shape}")
