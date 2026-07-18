"""Image preprocessing shared by serving and the optimization/research scripts.

Lives in serving/ because production code depends on it directly; the
research scripts import it from here instead of duplicating the transform.
"""
import torchvision.transforms as T

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def make_transform(size: int):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
