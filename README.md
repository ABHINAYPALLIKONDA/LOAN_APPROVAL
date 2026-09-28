# Loan Approval Prediction using Logistic Regression

## 📌 Project Overview

This project uses **Logistic Regression** to predict whether a loan application will be **Approved or Rejected** based on the given input features.

The project demonstrates an end-to-end **Machine Learning Classification workflow**, including:

* Data loading
* Missing value checking
* Label Encoding
* Train-Test Split
* Feature Scaling
* Logistic Regression
* Probability Prediction
* Confusion Matrix
* Classification Evaluation Metrics
* Data Visualization

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```text
Loan-Approval-Logistic-Regression/
│
├── loan_approval.csv
├── loan_approval.py
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The dataset used in this project is:

```text
loan_approval.csv
```

The first four columns are used as **input features (X)** and the fifth column is used as the **target variable (y)**.

```python
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]
```

The target variable is converted into numerical values using `LabelEncoder`.

---

## 🔄 Machine Learning Workflow

### 1. Load the Dataset

The dataset is loaded using Pandas.

```python
data = pd.read_csv("loan_approval.csv")
```

### 2. Check Missing Values

Missing values are checked using:

```python
data.isnull().sum()
```

### 3. Separate Features and Target

```python
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]
```

### 4. Encode the Target

The categorical target values are converted into numerical values.

```python
encoder = LabelEncoder()
y = encoder.fit_transform(y)
```

### 5. Train-Test Split

The dataset is divided into training and testing data.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.5,
    random_state=82
)
```

### 6. Feature Scaling

`StandardScaler` is used to standardize the input features.

```python
scaler = StandardScaler()

x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)
```

The scaler is fitted only on the training data and then used to transform the test data.

### 7. Logistic Regression Model

A Logistic Regression model with **L1 regularization** is trained.

```python
model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=1000
)

model.fit(x_train_scaler, y_train)
```

### 8. Prediction

The trained model predicts the loan approval class.

```python
prediction = model.predict(x_test_scaler)
```

### 9. Probability Prediction

The probability of each class is calculated using:

```python
pred_prob = model.predict_proba(x_test_scaler)
```

This helps understand how confident the model is about its predictions.

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate the classification model.

### Confusion Matrix

```python
confusion_matrix(y_test, prediction)
```

The confusion matrix shows:

* True Positive
* True Negative
* False Positive
* False Negative

### Accuracy

Measures the overall percentage of correct predictions.

```python
accuracy_score(y_test, prediction)
```

### Precision

Measures how many predicted positive cases were actually positive.

```python
precision_score(y_test, prediction)
```

### Recall

Measures how many actual positive cases were correctly identified.

```python
recall_score(y_test, prediction)
```

### F1 Score

The F1 score combines Precision and Recall.

```python
f1_score(y_test, prediction)
```

### Log Loss

Measures the quality of the predicted probabilities.

```python
log_loss(y_test, pred_prob)
```

Lower Log Loss generally indicates better probability estimates.

### ROC-AUC Score

Measures the model's ability to distinguish between the two classes.

```python
roc_auc_score(y_test, pred_prob[:, 1])
```

---

## 📊 Visualizations

### Actual vs Predicted

The project visualizes the actual class values against the predicted class values.

```python
plt.scatter(y_test, prediction)
```

This provides a simple visual comparison between actual and predicted classifications.

### Logistic Regression Probability

The probability of loan approval for each test sample is plotted.

A threshold line at **0.5** is also displayed.

```python
plt.axhline(
    y=0.5,
    linestyle="--"
)
```

The basic classification idea is:

```text
Probability < 0.5  → Class 0
Probability ≥ 0.5  → Class 1
```

The exact class meaning depends on how the target labels were encoded.

---

## 🧠 Concepts Demonstrated

This project helps demonstrate the following Machine Learning concepts:

* Classification
* Logistic Regression
* Binary Classification
* Label Encoding
* Train-Test Split
* Feature Scaling
* L1 Regularization
* `predict()`
* `predict_proba()`
* Classification Threshold
* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* Log Loss
* ROC-AUC
* Data Visualization

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project folder

```bash
cd Loan-Approval-Logistic-Regression
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python file

```bash
python loan_approval.py
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
matplotlib
scikit-learn
```

---

## 🎯 Project Objective

The main objective of this project is to understand how **Logistic Regression can be used for binary classification** and how to evaluate the performance of a classification model using multiple evaluation metrics.

---

## 👨‍💻 Author

**Abhinay**

This project is part of my **Machine Learning learning journey**, focusing on practical implementation of classification algorithms using Scikit-learn.
