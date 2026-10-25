"""Train the pneumonia classifier. Saves the best checkpoint by val accuracy."""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output", default="outputs/model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = build_model(pretrained=True, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = out_path.parent / "training_history.csv"

    best_acc, history = 0.0, []
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device, False)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            print(f"Epoch {epoch + 1}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(), "classes": classes, "val_acc": best_acc},
                           out_path)
                print(f"  -> saved best checkpoint to {out_path} (val acc {best_acc:.4f})")
    print(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
