# Demo run — pipeline test only, NOT real chest X-rays

This file records a test run made to prove the code works end-to-end.
It is **not** a result on real pneumonia data and must not be quoted as
pneumonia-detection accuracy.

## What was run

- Synthetic demo images were generated locally (dark background, two lung-like
  ellipses; PNEUMONIA images added bright white opacity blobs). These are
  computer-generated patterns, not patient X-rays.
- Split: train 24 per class, val 8 per class, test 8 per class.
- Model: ResNet18 transfer learning (ImageNet weights), 3 epochs, batch 8, CPU.
- Commands:
  - `python src/train.py --data-dir demo_data --epochs 3 --batch-size 8`
  - `python src/evaluate.py --data-dir demo_data --checkpoint outputs/demo_model.pth`
  - `python src/predict.py --image demo_data/test/PNEUMONIA/pneumonia_0.jpg`

## Demo results (synthetic data only)

- Validation accuracy: 1.0000 (3 epochs, synthetic patterns are easy to separate)
- Test accuracy: 1.0, precision/recall/F1: 1.0 for both classes, 16 test images
- Predict demo: PNEUMONIA image correctly predicted as PNEUMONIA, Grad-CAM overlay saved

These numbers only show that train → evaluate → Grad-CAM runs without errors.
They say nothing about performance on real chest X-rays.

## Next step for real results

Download the real Chest X-Ray Images (Pneumonia) dataset, organise it as
`data/train|val|test/NORMAL|PNEUMONIA`, run the same three commands with
`--data-dir data`, and report only the metrics from `evaluate.py` on that
real test split.
