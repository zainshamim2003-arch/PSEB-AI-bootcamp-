# Project 5 – Experiment Report

## 1. Objective
Build a live dashboard that plots training and validation loss and compares model experiments.

## 2. Dataset and split
- Dataset: Wisconsin Breast Cancer (scikit-learn)
- Rows: 569
- Features: 30 numeric measurements
- Split: 60% training / 20% validation / 20% test, stratified
- Scaling: fitted on training rows only

## 3. Experiments
Fill this table using the dashboard's downloaded CSV.

| Run | Hidden neurons | Dropout | Weight decay | Early stopping | Scaling | Diagnosis | Best validation loss | Malignant found | Missed | False alarms |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 1 | | | | | | | | | | |
| 2 | | | | | | | | | | |
| 3 | | | | | | | | | | |
| 4 | | | | | | | | | | |

## 4. Recommendation
Which experiment gave the most useful balance of validation behaviour and class-wise results? Explain why.

## 5. Limitations
1. Small, built-in educational dataset.
2. Simplified neural network and fixed 0.5 threshold.
3. Not clinically validated; not for use with real patients.
