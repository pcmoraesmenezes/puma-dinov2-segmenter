import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, target_size=(1024, 1024)):
        super().__init__()
        self.target_size = target_size
        
        # Refinamento de features antes do upsampling
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Camada final de predição
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 384, 74, 74)
        x = self.refiner(x) # -> (B, 128, 74, 74)
        
        # Upsampling para o tamanho da imagem original
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False) # -> (B, 128, 1024, 1024)
        
        return self.classifier(x) # -> (B, num_classes, 1024, 1024)

class DinoPUMASegmenter(nn.Module):
    def __init__(self, in_channels=384, tissue_classes=6, nuclei_classes=4):
        super().__init__()
        
        # Cabeça para Segmentação de Tecido (Macro)
        self.tissue_head = SegmentationHead(in_channels, tissue_classes)
        
        # Cabeça para Segmentação de Núcleos (Micro)
        self.nuclei_head = SegmentationHead(in_channels, nuclei_classes)

    def forward(self, x):
        # x: Tensores de features extraídos do DinoV2 (B, 384, 74, 74)
        tissue_logits = self.tissue_head(x)
        nuclei_logits = self.nuclei_head(x)
        
        return {
            "tissue": tissue_logits,
            "nuclei": nuclei_logits
        }

if __name__ == "__main__":
    # Teste de Sanidade (Forward Pass)
    model = DinoPUMASegmenter()
    dummy_input = torch.randn(1, 384, 74, 74)
    
    outputs = model(dummy_input)
    
    print(f"--- Teste de Arquitetura ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Tissue Output Shape: {outputs['tissue'].shape}") # Esperado: (1, 6, 1024, 1024)
    print(f"Nuclei Output Shape: {outputs['nuclei'].shape}") # Esperado: (1, 4, 1024, 1024)
    
    # Contagem de Parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros treináveis: {total_params / 1e6:.2f}M")
