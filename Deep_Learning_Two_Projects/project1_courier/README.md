# Project 1 – Little Data, Big Model

## Problem statement

A courier company has only 200 labelled images of postcode digits. Build a digit reader and use training/validation curves to check whether the network learns or memorises.

## Dataset and split

- Dataset: `sklearn.datasets.load_digits`
- Each image: 8 × 8 grayscale pixels, represented by 64 input values.
- Pixel range is scaled from 0–16 to 0–1.
- Fixed split: 200 training, 798 validation, 799 test examples.
- Random seed: 0.

## Models

| Model | Architecture | Optimisation |
|---|---|---|
| Small | 64 → 10 | SGD, learning rate 0.01 |
| Big | 64 → 256 → 256 → 10 | Adam, learning rate 0.001 |
| Big + regularisation | Same big network with Dropout(0.3) after hidden layers | AdamW, learning rate 0.001, weight decay 0.05, patience 30 |

All models use ReLU between hidden layers and cross-entropy loss.

## Run

```bash
pip install -r requirements.txt
python train.py
```

## Output files

- `outputs/curves.png` – all three training/validation curve pairs.
- `outputs/results.csv` – final training loss, best validation loss/epoch, epochs run and test accuracy.
- `outputs/best_*.pt` – best validation checkpoint for each model.
- `outputs/*_final.pt` – final evaluated model weights (best checkpoint loaded before saving).
- `outputs/config.json` – seed, device, hyperparameters and split.
- `outputs/classification_report.txt` – class-wise report for the regularised model.

## How to interpret the results

- **Underfitting:** the small model is too simple; training and validation losses remain relatively high.
- **Overfitting:** the big model can drive training loss very low while validation loss stops improving or rises.
- **Regularisation:** dropout discourages reliance on particular neurons; weight decay discourages unnecessarily large weights.
- **Early stopping:** stop after 30 epochs without a validation-loss improvement and retain the best checkpoint.

The course slide lists reference test accuracies of approximately 0.751, 0.934 and 0.937. Your run may differ slightly because of software versions and training details. Report the values produced by your own `results.csv`.
