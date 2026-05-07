import cv2
import numpy as np
import os

mask_path = "/home/paulo/Área de Trabalho/repo-pessoal/puma-dinov2-segmenter/puma/masks/nuclei/training_set_metastatic_roi_001.png"

if os.path.exists(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    unique_values = np.unique(mask)
    print(f"--- Diagnóstico de Máscara ---")
    print(f"Arquivo: {os.path.basename(mask_path)}")
    print(f"Valores únicos encontrados (Classes): {unique_values}")
    print(f"Pixels não-zero: {np.count_nonzero(mask)}")
    
    # Se quiser ver algo, multiplicamos por 50 e salvamos um "debug"
    debug_mask = mask * 40
    cv2.imwrite("puma-dinov2-segmenter/visualizations/debug_mask_visible.png", debug_mask)
    print(f"✅ Versão visível salva em: visualizations/debug_mask_visible.png")
else:
    print(f"Arquivo não encontrado: {mask_path}")
