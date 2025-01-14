# Time Series Air Quality Prediction with Neural Networks

## Overview
This project focuses on predicting air quality using time-series data. The core challenge of this project lies in **preprocessing the dataset**, specifically handling missing data and selecting the most informative features.

The dataset used in this project contains significant missing values (tagged as `-200`), and these gaps, particularly in target variables, posed unique challenges for model development. Advanced imputation techniques were applied, and experiments were conducted to evaluate the impact of different preprocessing strategies on model performance.

The project is divided into two tasks:
1. **Classification**: Predicting whether carbon monoxide (CO) levels exceed a threshold.
2. **Regression**: Predicting numeric nitrogen oxides (NOx) levels.

Both tasks use neural network models trained on rigorously preprocessed data, with model performance evaluated through a variety of metrics.


---

## Dataset
**`AirQualityUCI_Modified.xlsx`**
- **Source**: UCI Machine Learning Repository.
- **Description**:
  - Contains one year of hourly measurements (March 2004 to February 2005) collected from five metal oxide chemical sensors.
  - Includes pollutant concentrations such as carbon monoxide (CO), benzene, and nitrogen oxides (NOx).
  - Dataset was **modified** and pre-processed specifically for this project. Missing values are marked as `-200` in the raw data.

---

## Project Components

### 1. **Data Preprocessing**
**Key emphasis on handling missing data and selecting the right features:**
- Missing data filled using **k-Nearest Neighbors (kNN)** imputation.
- Features normalized using **Min-Max Scaling** to ensure compatibility with neural networks.
- **Feature Selection Strategies**:
  - **Mutual Information (Classification)**: To identify features reducing target uncertainty.
  - **Mutual Information Regression** (Regression): For continuous target dependency.
  - **Correlation Analysis**: To eliminate redundant or low-informative features.
    - Final features selected based on rigorous experimentation and evaluation of variable distributions.

### 2. **Classification Model**
- **Goal**: Predict whether carbon monoxide (CO) levels exceed a certain threshold.
- **Model Architecture**:
  - Input Layer (10 features).
  - 2 Fully Connected Hidden Layers (16 neurons each; ReLU activation).
  - 2 Dropout Layers (12% and 13% rates).
  - Output Layer (1 neuron; Sigmoid activation).
  - Optimized with Adam optimizer and binary cross-entropy loss.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score.
  - Confusion Matrix for performance visualization.

### 3. **Regression Model**
- **Goal**: Predict numeric nitrogen oxides (NOx) levels.
- **Model Architecture**:
  - Input Layer (10 features).
  - 2 Fully Connected Hidden Layers (16 neurons each; ReLU activation).
  - Dropout Layers with rates of 3% and 9.5%.
  - Output Layer (1 neuron; Linear activation).
  - Optimized with Adam optimizer and mean squared error loss.
- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE).
  - Mean Absolute Error (MAE).

---

## Results

### **Classification**:
- **Accuracy**: 94.46%
- **Precision**: 93.25%
- **Recall**: 93.08%
- **F1-Score**: 93.16%
- **Confusion Matrix**:
  - Visualizes the predicted vs actual proportions.

### **Regression**:
- **Root Mean Squared Error (RMSE)**: 61.47
- **Mean Absolute Error (MAE)**: 37.97

---

## Requirements
### Python Environment:
- Python 3.9+

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

---

## Usage
1. **Pre-trained Models**:
   - Place the data file `AirQualityUCI_Modified.xlsx` in the appropriate directory.
   - Load either the classification or regression model (`model_classification.h5`, `model_regression.h5`) for predictions.

2. **Training the Models**:
   - Open the notebook (`Time_Series_Air_Quality_Prediction.ipynb`).
   - Follow the guided steps to preprocess, train, and evaluate the models on the modified dataset.

3. **Visualize Model Performance**:
   - Use the notebook to generate plots for:
     - Loss and accuracy curves over training epochs.
     - Confusion Matrix (classification).
     - Actual vs Predicted graphs (regression).

---

## Acknowledgements
- Dataset sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- Feature Engineering inspired by resources on data imputation, normalization, and mutual information for feature selection.

---

## Notes
- Pre-trained models can be fine-tuned or directly utilized for prediction tasks.
- Always ensure the dataset follows the preprocessing structure specified in the notebook.

---

## License
MIT License. Feel free to use and modify the code, but cite the repository appropriately.

