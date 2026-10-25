"""Evaluate a checkpoint on the test split: accuracy, precision, recall, F1, ROC-AUC."""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from dataset import get_dataloaders
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = get_dataloaders(args.data_dir, batch_size=32)
    model = build_model(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loaders["test"]:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(probs.argmax(1))
            y_prob.extend(probs[:, 1])  # PNEUMONIA = class 1

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=classes, output_dict=True),
        "classes": classes,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
