import zipfile
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

BASE_PATH = "puma"
ROIS_ZIP = os.path.join(BASE_PATH, "ROIs.zip")
NUCLEI_ZIP = os.path.join(BASE_PATH, "nuclei.geojson.zip")
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_poly_coords(geom):
    """Extrai coordenadas tratando Polygon e MultiPolygon"""
    if geom['type'] == 'Polygon':
        return [geom['coordinates'][0]]
    elif geom['type'] == 'MultiPolygon':
        return [poly[0] for poly in geom['coordinates']]
    return []

def plot_sample(image_path):
    image_filename = os.path.basename(image_path)
    image_id = image_filename.replace(".tif", "")
    print(f"\n--- Diagnóstico: {image_id} ---")
    
    with zipfile.ZipFile(ROIS_ZIP, 'r') as z:
        with z.open(image_path) as f:
            img = Image.open(io.BytesIO(f.read()))
            img_np = np.array(img)
            print(f"Image resolution: {img_np.shape[1]}x{img_np.shape[0]}")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(img)
    
    geojson_filename = f"{image_id}_nuclei.geojson"
    
    try:
        with zipfile.ZipFile(NUCLEI_ZIP, 'r') as z:
            with z.open(geojson_filename) as f:
                data = json.load(f)
                
        features = data.get('features', [])
        all_coords = []
        count = 0
        
        for feat in features:
            polygons = get_poly_coords(feat['geometry'])
            for coords in polygons:
                all_coords.extend(coords)
                # Plotting with a thin line to see the real density
                poly = patches.Polygon(coords, linewidth=0.5, edgecolor='lime', facecolor='none', alpha=0.8)
                ax.add_patch(poly)
                count += 1
        
        # Coordinate check
        if all_coords:
            all_coords = np.array(all_coords)
            min_x, min_y = all_coords.min(axis=0)
            max_x, max_y = all_coords.max(axis=0)
            print(f"Range X no GeoJSON: {min_x:.1f} a {max_x:.1f}")
            print(f"Range Y no GeoJSON: {min_y:.1f} a {max_y:.1f}")
            
            if max_x > img_np.shape[1] or max_y > img_np.shape[0]:
                print("⚠️ WARNING: GeoJSON coordinates exceed the image size!")

        plt.title(f"{image_id} | Nuclei: {count}")
    except Exception as e:
        print(f"❌ Erro ao processar GeoJSON: {e}")

    plt.axis('off')
    output_path = os.path.join(OUTPUT_DIR, f"{image_id}_diag.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Diagnóstico salvo em: {output_path}")

if __name__ == "__main__":
    with zipfile.ZipFile(ROIS_ZIP, 'r') as z:
        tifs = [f for f in z.namelist() if f.endswith('.tif')]
    if tifs:
        for sample in tifs[:3]:
            plot_sample(sample)
