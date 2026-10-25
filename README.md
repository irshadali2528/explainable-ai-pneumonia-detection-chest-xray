# Explainable AI for Pneumonia Detection Using Chest X-ray Images

A deep learning project for chest X-ray image classification, built with transfer learning and explained with Grad-CAM.

- **Task:** Binary classification — `NORMAL` vs `PNEUMONIA` from chest X-ray images.
- **Model:** Transfer learning with a pretrained ResNet18 (ImageNet weights), with the final classifier replaced for 2 classes. The backbone can be partially frozen and then fine-tuned.
- **Explainability:** Grad-CAM on the last convolutional layer, visualising the image regions that most influenced each prediction, overlaid on the original X-ray.
- **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and a classification report (scikit-learn).

> **No accuracy is claimed in this README until training is run on a real dataset.** Run `train.py` on the dataset below and the metrics printed by `evaluate.py` are the results to report. Do not copy a number from another project or paper into this README.

## Dataset

This code expects the widely used Chest X-Ray Images (Pneumonia) dataset, organised as:

```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

The dataset itself is **not included** in this repository (size and licensing). Download it from its original Kaggle source, unzip it, and point `--data-dir` at the folder containing `train/`, `val/`, and `test/`.

Images are resized to 224×224, converted to 3 channels, and normalised with ImageNet mean/std, matching the pretrained backbone.

## Quick start

```bash
pip install -r requirements.txt

# Train (example: 10 epochs, ResNet18, batch 32)
python src/train.py --data-dir data --epochs 10 --batch-size 32 --output outputs/model.pth

# Evaluate on the test split and save a classification report + confusion matrix
python src/evaluate.py --data-dir data --checkpoint outputs/model.pth --output outputs/metrics.json

# Explain one image with Grad-CAM
python src/predict.py --image path/to/chest_xray.jpeg --checkpoint outputs/model.pth --output outputs/gradcam.png
```

`train.py` saves the best checkpoint by validation accuracy, plus a training-history CSV in `outputs/`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dataset.py      # ImageFolder datasets, transforms, data loaders
│   ├── model.py        # ResNet18 transfer-learning model
│   ├── gradcam.py      # Grad-CAM implementation (hooks, heatmap, overlay)
│   ├── train.py        # Training loop, validation, checkpointing
│   ├── evaluate.py     # Test metrics, confusion matrix, classification report
│   └── predict.py      # Single-image prediction + Grad-CAM overlay
└── outputs/            # Checkpoints, metrics and Grad-CAM images (git-ignored, except .gitkeep)
```

## How Grad-CAM works here

1. A forward pass records activations from the last convolutional block (`layer4` in ResNet18).
2. A backward pass for the predicted class gives gradients for those activations.
3. Gradients are globally averaged to weight each feature map, producing a coarse heatmap.
4. The heatmap is upsampled to 224×224, normalised, and overlaid on the original image with a colour map.

This shows *where* the model looked, not *why* in a clinical sense. A Grad-CAM overlay is an explanation aid for a clinician or reviewer, not a diagnosis on its own.

## Technologies

Python, PyTorch, Computer Vision, Deep Learning, Explainable AI (Grad-CAM), scikit-learn, Matplotlib, Pillow.
