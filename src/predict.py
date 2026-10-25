"""Predict one chest X-ray and save a Grad-CAM overlay."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, overlay_heatmap
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/gradcam.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pil_image = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    heatmap, pred_class = gradcam(input_tensor)
    overlay = overlay_heatmap(pil_image, heatmap)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    print(f"Prediction: {CLASSES[pred_class]}")
    print(f"Grad-CAM overlay saved to {out_path}")


if __name__ == "__main__":
    main()
