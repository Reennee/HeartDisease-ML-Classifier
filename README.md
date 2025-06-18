# Heart Disease Prediction with Optimization Techniques

## Project Overview
This project explores different optimization techniques in machine learning models for heart disease prediction. We compare classical ML algorithms (Logistic Regression, SVM, Random Forest) with neural networks using various optimization approaches (different optimizers, regularization techniques, dropout, early stopping).

## Dataset
The dataset contains 303 samples with 13 features related to heart health and a binary target variable indicating presence (1) or absence (0) of heart disease. Features include age, sex, chest pain type, blood pressure, cholesterol levels, etc.

## Results Summary

### Comparison Table

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Learning Rate | Dropout | Accuracy | F1 Score | Precision | Recall | ROC AUC | Loss |
|----------|-----------|-------------|--------|-----------------|---------------|---------|----------|----------|-----------|--------|---------|------|
| 1        | Adam      | None        | 100    | No              | 0.001         | 0.0     | 0.82     | 0.83     | 0.85      | 0.81   | 0.89    | 0.45 |
| 2        | Adam      | L2          | 50     | Yes             | 0.001         | 0.0     | 0.87     | 0.88     | 0.89      | 0.87   | 0.93    | 0.32 |
| 3        | RMSprop   | None        | 50     | Yes             | 0.0005        | 0.3     | 0.84     | 0.85     | 0.86      | 0.84   | 0.91    | 0.38 |
| 4        | Adam      | L1          | 50     | Yes             | 0.0001        | 0.2     | 0.86     | 0.87     | 0.88      | 0.86   | 0.92    | 0.35 |

### Best Performing Model
The best performing neural network was Instance 2 with:
- Adam optimizer
- L2 regularization (λ=0.01)
- Early stopping
- Learning rate of 0.001
- No dropout

This achieved 87% accuracy and 0.93 ROC AUC on the validation set.

### Classical ML vs Neural Network
The best classical ML model was Random Forest with hyperparameter tuning (n_estimators=100, max_depth=5) which achieved 85% accuracy. The optimized neural network slightly outperformed this with 87% accuracy, showing that with proper optimization techniques, neural networks can achieve better performance even on relatively small datasets.

## How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Summative_Intro_to_ml_Rene_Ntabana_assignment.ipynb`

Requirements:
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn