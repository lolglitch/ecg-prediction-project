# ECG Heart Disease Classification Using MLP

This project classifies ECG images into four classes using a Multi-Layer Perceptron neural network.

## Dataset

The dataset used is the ECG Image Dataset from Kaggle.

Classes:
- Myocardial Infarction
- History of MI
- Abnormal Heartbeat
- Normal

## Model

The model used is an MLP neural network. The ECG images are resized to 128x128, normalized, flattened, and passed through Dense layers.

## Files

- app.py: Streamlit application
- ecg_mlp_model.keras: trained model
- ECG_Classification_Notebook.ipynb: model training notebook
- requirements.txt: required libraries