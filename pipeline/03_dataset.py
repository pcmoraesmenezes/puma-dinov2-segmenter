import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


class PUMAFeatureDataset(Dataset):
    def __init__(self, base_path, feature_dir_name="features", cache_in_ram=True):
        self.base_path = base_path
        self.feature_dir = os.path.join(base_path, feature_dir_name)
        self.mask_tissue_dir = os.path.join(base_path, "masks/tissue")
        self.mask_nuclei_dir = os.path.join(base_path, "masks/nuclei")
        self.cache_in_ram = cache_in_ram
        
        # List of IDs based on the feature files
        self.sample_ids = sorted([f.replace(".pt", "") for f in os.listdir(self.feature_dir) if f.endswith(".pt")])
        
        if self.cache_in_ram:
            logger.info(f"Loading {len(self.sample_ids)} samples into RAM...")
            self.cached_features = []
            self.cached_tissue = []
            self.cached_nuclei = []
            for sample_id in self.sample_ids:
                feature_path = os.path.join(self.feature_dir, f"{sample_id}.pt")
                features = torch.load(feature_path).squeeze(0)
                
                tissue_mask_path = os.path.join(self.mask_tissue_dir, f"{sample_id}.png")
                nuclei_mask_path = os.path.join(self.mask_nuclei_dir, f"{sample_id}.png")
                
                tissue_mask = cv2.imread(tissue_mask_path, cv2.IMREAD_UNCHANGED)
                nuclei_mask = cv2.imread(nuclei_mask_path, cv2.IMREAD_UNCHANGED)
                
                tissue_mask = torch.from_numpy(tissue_mask).long()
                nuclei_mask = torch.from_numpy(nuclei_mask).long()
                
                self.cached_features.append(features)
                self.cached_tissue.append(tissue_mask)
                self.cached_nuclei.append(nuclei_mask)
            logger.info("RAM cache complete.")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        if self.cache_in_ram:
            features = self.cached_features[idx]
            tissue_mask = self.cached_tissue[idx]
            nuclei_mask = self.cached_nuclei[idx]
        else:
            # 1. Load features (C, H, W) -> (384, 74, 74)
            feature_path = os.path.join(self.feature_dir, f"{sample_id}.pt")
            features = torch.load(feature_path).squeeze(0) # Remove batch dim if present
            
            # 2. Load masks
            tissue_mask_path = os.path.join(self.mask_tissue_dir, f"{sample_id}.png")
            nuclei_mask_path = os.path.join(self.mask_nuclei_dir, f"{sample_id}.png")

            # Masks are saved at (1024, 1024)
            tissue_mask = cv2.imread(tissue_mask_path, cv2.IMREAD_UNCHANGED)
            nuclei_mask = cv2.imread(nuclei_mask_path, cv2.IMREAD_UNCHANGED)

            # Convert to Long tensors (classification labels)
            tissue_mask = torch.from_numpy(tissue_mask).long()
            nuclei_mask = torch.from_numpy(nuclei_mask).long()
        
        return {
            "id": sample_id,
            "features": features,
            "mask_tissue": tissue_mask,
            "mask_nuclei": nuclei_mask
        }
        

if __name__ == "__main__":
    # Quick test
    from config import BASE_PATH
    dataset = PUMAFeatureDataset(BASE_PATH, cache_in_ram=True)
    logger.info(f"Dataset initialized with {len(dataset)} samples.")

    sample = dataset[0]
    logger.debug(f"Sample ID: {sample['id']}")
    logger.debug(f"Features Shape: {sample['features'].shape}")
    logger.debug(f"Tissue Mask Shape: {sample['mask_tissue'].shape}")
    logger.debug(f"Nuclei Mask Shape: {sample['mask_nuclei'].shape}")
