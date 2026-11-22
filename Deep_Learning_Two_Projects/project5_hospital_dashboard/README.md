# Project 5 – Live Training Dashboard (City Hospital)

## Problem statement

A hospital data team wants a dashboard to inspect training and validation loss while a model is trained. The dashboard should help students see underfitting, overfitting and healthier learning curves, compare experiments, and inspect class-wise outcomes.

## Dataset – important clarification

The PowerPoint describes the Wisconsin Breast Cancer dataset as 569 patients with 30 measurements. In scikit-learn this is a **tabular dataset of numeric measurements**, not a collection of medical scan images.

- 569 rows
- 30 numeric input features
- Target is converted so `1 = malignant`, `0 = benign`
- Stratified split: 60% train, 20% validation, 20% test
- Seed: 0
- StandardScaler is fitted on training rows only, then applied to validation and test rows.

## Dashboard features

- Sidebar controls for hidden-layer size, dropout, weight decay, learning rate, epochs, early stopping, patience and scaling.
- Training and validation loss chart refreshed during training (every five epochs).
- Curve diagnosis using the simple rules from the course slide:
  - training loss above 0.3 → underfitting
  - final validation loss more than 30% above its best → overfitting
  - otherwise → healthy
- Validation counts: malignant cases found, malignant cases missed, and false alarms.
- Experiment history table and CSV download.
- Separate one-time final test button.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested experiment sequence

Run at least four experiments, changing one setting at a time:

1. 256 neurons, dropout 0, weight decay 0, early stopping off.
2. Keep the same model and add dropout 0.3 and weight decay 0.05.
3. Keep those settings and enable early stopping.
4. Reduce the model to 16 neurons while keeping the regularisation settings.
5. Optional: turn scaling off and compare curve stability.

Record the diagnosis, best validation loss, malignant cases found, missed cases and false alarms. The best configuration is not automatically the one with the highest accuracy: examine missed positive cases and false alarms as well.

## Safety and limitations

This is a classroom coding exercise, not a validated medical model or medical device. It must not be used to diagnose, screen, triage, or make decisions about real patients. The dataset is small and historical; performance on this dataset does not establish clinical safety, fairness, or generalisation. A qualified clinical team and extensive independent validation would be required for any real healthcare application.
