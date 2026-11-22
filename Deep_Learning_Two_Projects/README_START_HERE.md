# Deep Learning – Two Completed Course Projects

This package contains two independent educational projects based on **Part 3 – Train Better and Ship**.

## Projects included

1. **Project 1 – Little Data, Big Model (Courier Company)**
   - Uses the scikit-learn handwritten digits dataset (8×8 images).
   - Trains a small model, a larger model, and a regularised larger model.
   - Records training and validation loss, plots the three runs, evaluates the test set, and saves the best checkpoint with early stopping.

2. **Project 5 – Live Training Dashboard (City Hospital)**
   - Uses the Wisconsin Breast Cancer dataset bundled with scikit-learn.
   - Builds a Streamlit dashboard to compare training experiments.
   - Displays live training/validation curves, a simple curve diagnosis, evaluation counts, a run history table, and CSV downloads.
   - **Educational demonstration only:** this is not a medical device and must not be used to diagnose, screen, or make decisions about real people.

## Requirements

- Python 3.10 or newer
- Install packages separately inside each project folder.

## Start Project 1

```bash
cd project1_courier
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
```

Expected output files are created in `outputs/`, including `curves.png`, `results.csv`, and model checkpoints.

## Start Project 5

```bash
cd project5_hospital_dashboard
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens. Use the sidebar to choose model settings and click **Train experiment**. Run at least four experiments, changing one setting at a time. Download the experiment table as CSV.

## Important notes

- The datasets are built into scikit-learn; no separate dataset download is required.
- Results can differ slightly with package versions and hardware. The PowerPoint's figures are reference results, not a guarantee that every environment will reproduce identical decimals.
- Project 5's dataset contains 569 rows and 30 numeric measurements, not medical image scans. The dashboard is a classroom demonstration only.
