
# End-to-End Machine Learning Project: Diabetes Prediction

This project demonstrates a full machine learning workflow for predicting the presence of diabetes using the **Pima Indians Diabetes Dataset**.

## 📁 Dataset
- The dataset is sourced from Kaggle: [`diabetes.csv`](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- It includes medical data (like glucose level, BMI, age) for female patients of Pima Indian heritage.

## 🔧 Technologies Used
- Python 3 (Kaggle environment)
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

## 📊 Workflow
1. **Data Loading & Exploration**
   - Basic info, statistics, and initial data understanding.
2. **Preprocessing**
   - Handling missing values using `SimpleImputer`
   - Feature scaling with `StandardScaler`
3. **Model Training**
   - Multiple classification models: Logistic Regression, Random Forest, SVM, XGBoost, etc.
   - Hyperparameter tuning using `GridSearchCV`
4. **Evaluation**
   - Accuracy, Confusion Matrix, ROC-AUC
5. **Visualization**
   - Correlation heatmaps, feature importance plots, learning curves

## 🚀 How to Run
If running locally or in another Jupyter environment:
1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
2. Load the notebook:
```bash
jupyter notebook end-to-end-ml.ipynb
```
3. Make sure the dataset (`diabetes.csv`) is in the correct path.

## 📌 Notes
- The notebook is originally designed for Kaggle environment, so change data paths if running elsewhere.
- Warnings are suppressed for clean output.

## 📬 Contact
For any questions or suggestions, feel free to reach out.
