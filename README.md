# 🌲 Random Forest Classification Project

## 📌 Project Overview

This project implements a **Random Forest Classifier** to solve a classification problem using a structured Machine Learning workflow.

The objective of this project is to:

* Perform data preprocessing and feature engineering
* Train multiple classification models
* Compare model performance using proper evaluation metrics
* Tune hyperparameters for optimal performance
* Save the trained model for future use

This project demonstrates practical understanding of model building, evaluation, and deployment-ready saving techniques.

---

## 📂 Project Structure

Random_Forest_Project/
│
├── Data/                  # Dataset folders
│   ├── clean/             # Cleaned dataset
│   ├── processed/         # Feature engineered dataset
│   └── Raw/               # Original raw dataset
│
├── models/                # Saved trained models (.pkl)
│   ├── Decision_Tree.pkl
│   ├── Logistic_Regression.pkl
│   └── RandomForest.pkl
│
├── Notebook/               # Saved trained models (.pkl)
│   ├── 01_defining_the_problem.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Training.ipynb
│
├── venv/
├── .gitignore
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation


---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## 🧠 Model Used

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.

Key advantages:

* Handles non-linearity well
* Reduces variance using bagging
* Works well with both numerical and categorical features
* Robust to outliers

---

## 🔄 Workflow

### 1️⃣ Data Loading

The dataset is loaded using pandas and cleaned for further processing.

```python
df = pd.read_csv("Data/Raw/Travel.csv")
```

---

### 2️⃣ Feature Engineering

Created new features and removed unnecessary columns.

Example:

```python
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
```

---

### 3️⃣ Train-Test Split

The dataset is split into training and testing sets to evaluate model generalization.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=30
)
```

---

### 4️⃣ Model Training

Random Forest was trained with optimized parameters:

```python
RandomForestClassifier(
    min_samples_split=2,
    max_features=None,
    max_depth=None,
    criterion='log_loss'
)
```

---

### 5️⃣ Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

Example:

```python
accuracy_score(y_test, y_test_pred)
```

---

### 6️⃣ Model Saving

The trained model is saved using joblib for future predictions.

```python
joblib.dump(model, "models/RandomForest.pkl")
```

To load the model:

```python
model = joblib.load("models/RandomForest.pkl")
```

---

## 📊 Evaluation Metrics Explained

Accuracy
Measures overall correctness of the model.

Precision
Measures how many predicted positives were actually correct.

Recall
Measures how many actual positives were correctly predicted.

F1 Score
Harmonic mean of precision and recall.

ROC-AUC
Measures the model’s ability to distinguish between classes.

---

## 🚀 Key Learnings

* Importance of comparing multiple models
* Understanding overfitting by comparing train and test performance
* Importance of hyperparameter tuning
* Proper model saving and loading techniques
* Writing modular and production-ready ML code

---

## 📈 Future Improvements

* Implement GridSearchCV for hyperparameter tuning
* Add Cross-Validation
* Build a full Pipeline (Preprocessing + Model)
* Deploy model using Flask or FastAPI
* Create a simple prediction UI

---

## 👨‍💻 Author

Amaan Behlim
Machine Learning Enthusiast | AI/ML Student

