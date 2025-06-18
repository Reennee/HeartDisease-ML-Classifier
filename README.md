# Heart Disease Prediction with Optimization Techniques

## Project Overview
This project explores different optimization and regularization techniques in machine learning models for heart disease prediction. I compared classical ML algorithms (Logistic Regression, SVM, Random Forest) with neural networks using various optimization approaches (different optimizers, regularization techniques, dropout, early stopping). The workflow includes model training, evaluation, error analysis, and model persistence for reproducibility and deployment.

## Dataset
The dataset contains 303 samples with 13 features related to heart health and a binary target variable indicating presence (1) or absence (0) of heart disease. Features include age, sex, chest pain type, blood pressure, cholesterol levels, etc.

## Workflow Summary
1. **Data Loading & Preprocessing**: Data is loaded, split into train/validation/test sets, and standardized.
2. **Model Training**:
   - Classical ML models: Logistic Regression, SVM, Random Forest (with hyperparameter tuning)
   - Neural Network models: Multiple instances with different optimizers, regularization, dropout, and early stopping
3. **Model Evaluation & Error Analysis**:
   - Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
   - Visualizations: Confusion Matrix (with heatmap), ROC Curve
   - Classification Report: Precision, recall, f1-score per class
4. **Model Saving & Loading**:
   - All trained models are saved with unique filenames for reproducibility
   - Saved models can be loaded and used for predictions on new/test data
5. **Prediction & Reporting**:
   - Predictions are made using the best saved model (Instance 2)
   - Evaluation metrics and plots are generated for predictions on the test set

## Results Summary

### Comparison Table

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Learning Rate | Dropout | Accuracy | F1 Score | Precision | Recall | ROC AUC | Loss |
|----------|-----------|-------------|--------|-----------------|---------------|---------|----------|----------|-----------|--------|---------|------|
| 1        | Adam      | None        | 100    | No              | 0.001         | 0.0     | 0.95     | 0.96     | 0.92      | 1      | 0.96    | 0.22 |
| 2        | Adam      | L2          | 50     | Yes             | 0.001         | 0.0     | 0.91     | 0.92     | 0.88      | 0.95   | 0.93    | 0.27 |
| 3        | RMSprop   | None        | 50     | Yes             | 0.0005        | 0.3     | 0.91     | 0.82     | 0.88      | 0.95   | 0.91    | 0.28 |
| 4        | Adam      | L1          | 50     | Yes             | 0.0001        | 0.2     | 0.93     | 0.93     | 0.92      | 0.95   | 0.95    | 0.25 |

### Best Performing Model
The best performing neural network was Instance 1 with:
- Adam optimizer
-  regularization (None)
- Early stopping = No
- Learning rate of 0.001
- No dropout

This achieved 95% accuracy and 0.96 ROC AUC on the validation set.

### Classical ML vs Neural Network
The best classical ML model was Random Forest with hyperparameter tuning (n_estimators=100, max_depth=5) which achieved 88% accuracy. The optimized neural network outperformed this with 95% accuracy, showing that with proper optimization techniques, neural networks can achieve better performance even on relatively small datasets.

## Error Analysis & Model Evaluation
- **Confusion Matrix**: Visualized for each model to show true/false positives/negatives
- **Classification Report**: Precision, recall, f1-score for each class
- **ROC Curve**: Plotted for each model to visualize trade-off between sensitivity and specificity
- **Comprehensive Metrics**: All models are evaluated on accuracy, precision, recall, f1, and ROC AUC

## Model Saving & Loading
- All models (classical and neural networks) are saved with unique filenames after training
- Example for neural network: `model.save('nn_instance2.h5')`
- Example for classical model: `joblib.dump(rf, 'random_forest_model.pkl')`
- Models can be loaded using `load_model` (Keras) or `joblib.load` (scikit-learn) for future predictions

## Making Predictions with Saved Models
To make predictions with a saved model (e.g., Instance 2 neural network):
```python
from tensorflow.keras.models import load_model
model = load_model('nn_instance2.h5')
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
```

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
