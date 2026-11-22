"""Project 5: Live Training Dashboard – educational demonstration only."""
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from trainer import train, diagnose

st.set_page_config(page_title="Live Training Dashboard", page_icon="📈", layout="wide")
st.title("Project 5 – Live Training Dashboard")
st.caption("City Hospital scenario | Deep Learning course project")

st.warning(
    "Educational demonstration only. This model is not a medical device, "
    "is not clinically validated, and must not be used to diagnose or screen "
    "real people or guide healthcare decisions."
)

@st.cache_data
def load_and_split(scale_features=True):
    dataset = load_breast_cancer()
    X = dataset.data.astype(np.float32)
    # sklearn target: 0 = malignant, 1 = benign. Convert so malignant is class 1.
    y = (dataset.target == 0).astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    return (
        to_tensor(X_train), torch.tensor(y_train, dtype=torch.long),
        to_tensor(X_val), torch.tensor(y_val, dtype=torch.long),
        to_tensor(X_test), torch.tensor(y_test, dtype=torch.long),
    )

if "runs" not in st.session_state:
    st.session_state.runs = []
if "final_test_done" not in st.session_state:
    st.session_state.final_test_done = False

st.sidebar.header("Experiment settings")
hidden_size = st.sidebar.selectbox("Hidden neurons", [16, 32, 64, 256], index=0)
dropout = st.sidebar.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
weight_decay = st.sidebar.select_slider(
    "Weight decay", options=[0.0, 0.001, 0.01, 0.05, 0.1], value=0.05
)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001
)
epochs = st.sidebar.slider("Maximum epochs", 20, 300, 200, 10)
early_stopping = st.sidebar.checkbox("Enable early stopping", value=True)
patience = st.sidebar.slider("Patience", 5, 50, 20, 5)
scale_features = st.sidebar.checkbox("Scale features", value=True)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=0)

cfg = {
    "hidden_size": hidden_size,
    "dropout": dropout,
    "weight_decay": weight_decay,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "early_stopping": early_stopping,
    "patience": patience,
    "scale_features": scale_features,
    "seed": int(seed),
}

st.subheader("1. Train an experiment")
st.write(
    "The dataset has 569 rows and 30 numeric measurements. The split is "
    "stratified into 60% training, 20% validation and 20% test data. "
    "The scaler, when enabled, is fitted on training data only."
)

if st.button("Train experiment", type="primary"):
    data_all = load_and_split(scale_features)
    X_train, y_train, X_val, y_val, X_test, y_test = data_all
    progress = st.progress(0.0)
    chart_placeholder = st.empty()
    status = st.empty()

    def redraw(epoch, history):
        progress.progress(min(epoch / int(cfg["epochs"]), 1.0))
        if epoch % 5 == 0 or epoch == 1:
            frame = pd.DataFrame({
                "Training loss": history["train"],
                "Validation loss": history["val"],
            })
            chart_placeholder.line_chart(frame)
            status.caption(f"Training epoch {epoch}")

    model, history, best_val, best_epoch, epochs_run, device = train(
        cfg, (X_train, y_train, X_val, y_val), on_epoch=redraw
    )
    verdict = diagnose(history)

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_val.to(device)), dim=1)[:, 1]
        val_pred = (val_probs >= 0.5).long().cpu().numpy()
    y_val_np = y_val.numpy()
    cm = confusion_matrix(y_val_np, val_pred, labels=[0, 1])
    # labels: 0=benign, 1=malignant
    tn, fp, fn, tp = cm.ravel()

    record = {
        **cfg,
        "diagnosis": verdict,
        "epochs_run": epochs_run,
        "best_validation_loss": round(best_val, 5),
        "validation_accuracy": round(accuracy_score(y_val_np, val_pred), 4),
        "malignant_cases_found": int(tp),
        "malignant_cases_missed": int(fn),
        "false_alarms": int(fp),
    }
    st.session_state.runs.append(record)
    st.session_state.latest = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": history,
        "record": record,
        "confusion_matrix": cm.tolist(),
        "test_data": (X_test, y_test),
        "device": str(device),
    }
    progress.progress(1.0)
    st.success(f"Experiment complete. Curve diagnosis: {verdict}")
    st.write(f"Best validation loss: **{best_val:.4f}** at epoch **{best_epoch}**.")
    st.write(
        f"Validation counts — malignant found: **{tp}**, "
        f"malignant missed: **{fn}**, false alarms: **{fp}**."
    )
    st.caption("The validation threshold is 0.5 for this classroom exercise.")

st.subheader("2. Experiment history")
if st.session_state.runs:
    runs_df = pd.DataFrame(st.session_state.runs)
    st.dataframe(runs_df, use_container_width=True)
    csv_bytes = runs_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download experiments CSV",
        data=csv_bytes,
        file_name="hospital_experiments.csv",
        mime="text/csv",
    )
    if len(runs_df) < 4:
        st.info(f"Run at least {4 - len(runs_df)} more experiment(s) to meet the project requirement.")
else:
    st.info("No experiments yet. Configure the sidebar and click Train experiment.")

st.subheader("3. Final test – run once after choosing settings")
st.write(
    "The test set is held back from training and validation. Choose your settings "
    "and complete your experiments first. Then run the final test once."
)
if st.button("Run final test once", disabled=(not st.session_state.runs or st.session_state.final_test_done)):
    # Train the selected configuration, then evaluate once on the held-out test split.
    # The button is deliberately one-time per browser session.
    data_all = load_and_split(scale_features)
    X_train, y_train, X_val, y_val, X_test, y_test = data_all
    model, history, best_val, best_epoch, epochs_run, device = train(
        cfg, (X_train, y_train, X_val, y_val)
    )
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test.to(device)), dim=1)[:, 1]
        pred = (probs >= 0.5).long().cpu().numpy()
    y_test_np = y_test.numpy()
    cm_test = confusion_matrix(y_test_np, pred, labels=[0, 1])
    tn, fp, fn, tp = cm_test.ravel()
    st.session_state.final_test_done = True
    st.session_state.final_result = {
        "accuracy": accuracy_score(y_test_np, pred),
        "malignant_found": int(tp),
        "malignant_missed": int(fn),
        "false_alarms": int(fp),
        "confusion_matrix": cm_test.tolist(),
    }

if st.session_state.final_test_done:
    result = st.session_state.final_result
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{result['accuracy']:.3f}")
    c2.metric("Malignant found", result["malignant_found"])
    c3.metric("Malignant missed", result["malignant_missed"])
    c4.metric("False alarms", result["false_alarms"])
    st.write("Confusion matrix (rows = actual [benign, malignant], columns = predicted [benign, malignant]):")
    st.write(pd.DataFrame(
        result["confusion_matrix"],
        index=["Actual benign", "Actual malignant"],
        columns=["Predicted benign", "Predicted malignant"],
    ))

st.subheader("4. What the dashboard diagnosis means")
st.markdown(
    "- **Underfitting:** final training loss is above 0.3 (the course's simple rule).\n"
    "- **Overfitting:** final validation loss is more than 30% above its best value.\n"
    "- **Healthy curve:** neither rule is triggered.\n\n"
    "These are simplified teaching rules, not clinical quality criteria."
)
