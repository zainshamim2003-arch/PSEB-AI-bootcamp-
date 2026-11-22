# Data and label note

`sklearn.datasets.load_breast_cancer()` returns 569 rows and 30 numeric features. Its original target encoding is 0 for malignant and 1 for benign. The app reverses this into 1 for malignant and 0 for benign to match the course slide's convention.

The app uses a stratified 60/20/20 split. Standardisation is fitted only on the training partition to avoid leaking validation/test statistics into training.
