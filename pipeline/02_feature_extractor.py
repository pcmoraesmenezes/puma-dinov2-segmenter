import zipfile
import json
import os
import sys
import cv2
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
import torch
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import ROIS_ZIP, NUCLEI_ZIP, TISSUE_ZIP, MASK_TISSUE_DIR, MASK_NUCLEI_DIR, FEATURES_DIR

logger = get_logger(__name__)

OUTPUT_MASK_TISSUE = MASK_TISSUE_DIR
OUTPUT_MASK_NUCLEI = MASK_NUCLEI_DIR
OUTPUT_FEATURES = FEATURES_DIR

os.makedirs(OUTPUT_MASK_TISSUE, exist_ok=True)
os.makedirs(OUTPUT_MASK_NUCLEI, exist_ok=True)
os.makedirs(OUTPUT_FEATURES, exist_ok=True)

# Class Mapping
TISSUE_MAPPING = {
    'tissue_white_background': 0,
    'tissue_stroma': 1,
    'tissue_blood_vessel': 2,
    'tissue_tumor': 3,
    'tissue_epidermis': 4,
    'tissue_necrosis': 5
}

NUCLEI_MAPPING = {
    'nuclei_tumor': 1,
    'nuclei_lymphocyte': 2,
    'nuclei_plasma_cell': 2,
}

# --- DinoV2 Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Loading DINOv2 model
MODEL_NAME = "dinov2_vits14"
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
model.to(DEVICE)
model.eval()

# DinoV2 requires multiples of 14. 1024 isn't. 14 * 74 = 1036.
# We resize to 1036 to keep the resolution close.
TRANSFORM = T.Compose([
    T.Resize((1036, 1036)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Rasterization functions ---

def get_poly_coords(geom):
    if geom['type'] == 'Polygon':
        return [np.array(geom['coordinates'][0], dtype=np.int32)]
    elif geom['type'] == 'MultiPolygon':
        return [np.array(poly[0], dtype=np.int32) for poly in geom['coordinates']]
    return []

def create_mask(geojson_data, shape, mapping, is_nuclei=False):
    mask = np.zeros(shape, dtype=np.uint8)
    features = geojson_data.get('features', [])
    for feat in features:
        name = feat.get('properties', {}).get('classification', {}).get('name', '')
        if is_nuclei:
            class_id = NUCLEI_MAPPING.get(name, 3)
        else:
            class_id = TISSUE_MAPPING.get(name, 0)
        polygons = get_poly_coords(feat['geometry'])
        for poly in polygons:
            cv2.fillPoly(mask, [poly], class_id)
    return mask

# --- Loop Principal ---

def process_all():
    logger.info("Starting feature extraction and mask generation...")
    
    with zipfile.ZipFile(ROIS_ZIP, 'r') as z_rois, \
         zipfile.ZipFile(NUCLEI_ZIP, 'r') as z_nuclei, \
         zipfile.ZipFile(TISSUE_ZIP, 'r') as z_tissue:
        
        tifs = sorted([f for f in z_rois.namelist() if f.endswith('.tif')])
        
        for tif_path in tqdm(tifs, desc="Processando ROIs"):
            image_id = os.path.basename(tif_path).replace(".tif", "")
            
            # --- 1. Feature Extraction ---
            feature_path = os.path.join(OUTPUT_FEATURES, f"{image_id}.pt")
            
            if not os.path.exists(feature_path):
                with z_rois.open(tif_path) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    orig_width, orig_height = img.size
                    
                    input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        # DinoV2 forward_features returns a dict with 'x_norm_patchtokens' among others
                        # For segmentation, we want the patch tokens (B, L, C)
                        # Batch -> how many images are being processed simultaneously
                        # Length -> number of patches (74*74 = 5476)
                        # Channels -> embedding dimension (384 for vit-s/14)
                        
                        out = model.get_intermediate_layers(input_tensor, n=1)[0]
                        # Reshape from (1, 74*74, 384) to (1, 384, 74, 74)
                        b, l, c = out.shape
                        patch_h = patch_w = int(l**0.5)
                        features = out.reshape(b, patch_h, patch_w, c).permute(0, 3, 1, 2)
                        
                        torch.save(features.cpu(), feature_path)
                        logger.debug(f"Features saved: {feature_path}")
            else:
                logger.debug(f"Features already cached, skipping DINOv2: {image_id}")
                with z_rois.open(tif_path) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    orig_width, orig_height = img.size

            # --- 2. Mask Generation (if they don't exist) ---
            tissue_mask_path = os.path.join(OUTPUT_MASK_TISSUE, f"{image_id}.png")
            if not os.path.exists(tissue_mask_path):
                tissue_geojson = f"01_training_dataset_geojson_tissue/{image_id}_tissue.geojson"
                if tissue_geojson not in z_tissue.namelist():
                    tissue_geojson = f"{image_id}_tissue.geojson"
                if tissue_geojson in z_tissue.namelist():
                    with z_tissue.open(tissue_geojson) as f:
                        data = json.load(f)
                        mask = create_mask(data, (orig_height, orig_width), TISSUE_MAPPING)
                        cv2.imwrite(tissue_mask_path, mask)

            nuclei_mask_path = os.path.join(OUTPUT_MASK_NUCLEI, f"{image_id}.png")
            if not os.path.exists(nuclei_mask_path):
                nuclei_geojson = f"{image_id}_nuclei.geojson"
                if nuclei_geojson in z_nuclei.namelist():
                    with z_nuclei.open(nuclei_geojson) as f:
                        data = json.load(f)
                        mask = create_mask(data, (orig_height, orig_width), NUCLEI_MAPPING, is_nuclei=True)
                        cv2.imwrite(nuclei_mask_path, mask)

if __name__ == "__main__":
    process_all()
    logger.info("Extraction complete.")
