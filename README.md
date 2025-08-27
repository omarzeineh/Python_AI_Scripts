# python_AI_Scripts

A collection of Python scripts covering topics in Data Cleaning, Machine Learning, Artificial Intelligence, and Algorithm Visualization.  
This repository is intended for educational use and includes both from-scratch implementations and scikit-learn based models.

---

## 📂 Scripts Overview

### 1. `data_cleaning.py`
- Cleans and merges multiple datasets.
- Handles outliers using IQR.
- Imputes missing values with column means.
- Normalizes numerical features for machine learning models.
- Saves the cleaned dataset as CSV.

### 2. `regression_model_fitting.py`
- Loads the cleaned dataset.
- Fits **Random Forest Regressors** to predict:
  - **C** variable
  - **MET** variable
- Evaluates performance using **Root Mean Squared Error (RMSE)**.

### 3. `knights_tour_gui.py`
- Tkinter-based GUI for the **Knight’s Tour problem**.
- Supports chessboard sizes (5x5 up to 8x8).
- Adjustable animation speed.
- Highlights knight’s path with step-by-step visualization.

### 4. `knn_classifier.py`
- Implements **K-Nearest Neighbors (KNN)** from scratch.
- Uses Euclidean distance for classification.
- Includes confusion matrix, accuracy, recall, precision, F1 score.
- Example dataset included for demonstration.

### 5. `ml_classifiers_comparison.py`
- Compares different ML classifiers on the **Iris dataset**:
  - KNN
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - Linear SVM
  - Kernel SVM
- Prints test accuracy for each classifier.

### 6. `ml_hyperparameter_tuning.py`
- Performs **hyperparameter tuning** on synthetic datasets (moons, circles, blobs).
- Searches best parameters for:
  - KNN (k-value)
  - Decision Trees (max depth)
  - Random Forests (max depth & estimators)
  - SVM (C, Gamma, and kernel choice)

### 7. `fsm_motor_control.py`
- Implements a **Finite State Machine (FSM)** to simulate:
  - DC motor behavior
  - Warm-up and Active LED states
  - Button inputs (using `gpiozero` & Tkinter circuit simulator)
- Models Idle → Warm-up → Active transitions.

### 8. `polynomial_regression_plot.py`
- Fits **Polynomial Regression (degree=2)**.
- Evaluates with RMSE on train/test split.
- Plots predicted vs actual values.
- Includes curve visualization.

### 9. `polynomial_regression_fullfit.py`
- Fits Polynomial Regression on **entire dataset**.
- Evaluates overall RMSE.
- Plots full curve for better visualization.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/python_AI_Scripts.git
   cd python_AI_Scripts
