import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, target_size=(1024, 1024)):
        super().__init__()
        self.target_size = target_size
        
        # Feature refinement before upsampling
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction layer
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x, upscale=True):
        # x: (B, 384, 74, 74)
        x = self.refiner(x) # -> (B, 128, 74, 74)
        
        if upscale:
            # Upsampling to the original image size
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False) # -> (B, 128, 1024, 1024)
        
        return self.classifier(x) # -> (B, num_classes, H, W)

class DinoPUMASegmenter(nn.Module):
    def __init__(self, in_channels=384, tissue_classes=6, nuclei_classes=4):
        super().__init__()
        
        # Head for Tissue Segmentation (Macro)
        self.tissue_head = SegmentationHead(in_channels, tissue_classes)

        # Head for Nuclei Segmentation (Micro)
        self.nuclei_head = SegmentationHead(in_channels, nuclei_classes)

    def forward(self, x, upscale=True):
        # x: feature tensors extracted from DinoV2 (B, 384, 74, 74)
        tissue_logits = self.tissue_head(x, upscale=upscale)
        nuclei_logits = self.nuclei_head(x, upscale=upscale)
        
        return {
            "tissue": tissue_logits,
            "nuclei": nuclei_logits
        }

if __name__ == "__main__":
    # Sanity Test (Forward Pass)
    model = DinoPUMASegmenter()
    dummy_input = torch.randn(1, 384, 74, 74)

    outputs = model(dummy_input)

    print(f"--- Architecture Test ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Tissue Output Shape: {outputs['tissue'].shape}") # Expected: (1, 6, 1024, 1024)
    print(f"Nuclei Output Shape: {outputs['nuclei'].shape}") # Expected: (1, 4, 1024, 1024)

    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
