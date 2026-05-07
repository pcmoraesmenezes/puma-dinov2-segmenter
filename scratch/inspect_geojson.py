import zipfile
import json
import os

def inspect_zip(zip_path):
    print(f"\n--- Inspecting {zip_path} ---")
    if not os.path.exists(zip_path):
        print(f"Path does not exist: {zip_path}")
        return
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('.geojson')]
        if not files:
            print("No geojson files found.")
            return
        
        first_file = files[0]
        print(f"First file: {first_file}")
        with z.open(first_file) as f:
            data = json.load(f)
            features = data.get('features', [])
            print(f"Total features: {len(features)}")
            if features:
                print("Sample feature properties:")
                print(json.dumps(features[0]['properties'], indent=2))
                
                # Collect all unique classes (classification -> name)
                classes = set()
                for feat in features:
                    props = feat.get('properties', {})
                    classification = props.get('classification', {})
                    name = classification.get('name')
                    if name:
                        classes.add(name)
                print(f"Unique classes found: {classes}")

BASE = "/home/paulo/Área de Trabalho/repo-pessoal/puma-dinov2-segmenter/puma"
inspect_zip(os.path.join(BASE, "nuclei.geojson.zip"))
inspect_zip(os.path.join(BASE, "tissue.geojson.zip"))
