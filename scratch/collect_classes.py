import zipfile
import json
import os

def collect_all_classes(zip_path):
    print(f"\n--- Collecting all classes from {zip_path} ---")
    if not os.path.exists(zip_path):
        print(f"Path does not exist: {zip_path}")
        return
        
    all_classes = set()
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('.geojson')]
        for file_name in files:
            with z.open(file_name) as f:
                try:
                    data = json.load(f)
                    features = data.get('features', [])
                    for feat in features:
                        props = feat.get('properties', {})
                        classification = props.get('classification', {})
                        name = classification.get('name')
                        if name:
                            all_classes.add(name)
                except:
                    pass
    print(f"Total unique classes found: {all_classes}")

BASE = "/home/paulo/Área de Trabalho/repo-pessoal/puma-dinov2-segmenter/puma"
collect_all_classes(os.path.join(BASE, "nuclei.geojson.zip"))
collect_all_classes(os.path.join(BASE, "tissue.geojson.zip"))
