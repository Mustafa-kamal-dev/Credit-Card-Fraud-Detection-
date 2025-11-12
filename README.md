# 💳 Credit Card Fraud Detection using Machine Learning

Fraudulent transactions are a major challenge for financial institutions.  
In this project, I developed a **Machine Learning model** to accurately detect fraudulent credit card transactions using **Python, Logistic Regression, and data balancing techniques**.

---

## 🚀 Project Overview

The goal of this project is to predict whether a credit card transaction is **legit (0)** or **fraudulent (1)**.  
Since fraud cases are very rare, the dataset is **highly imbalanced**, so special sampling techniques were applied to balance the data before training.

---

## 🧠 Technologies Used

- **Python 3**
- **NumPy** – for numerical computation  
- **Pandas** – for data analysis and manipulation  
- **Scikit-learn** – for model training and evaluation  
- **Matplotlib** & **Seaborn** – for data visualization  

---

## 🧩 Project Workflow

1. **Data Loading & Exploration**  
   - Loaded the dataset and inspected basic info using `info()` and `describe()`  
   - Checked imbalance using `value_counts()` on the `Class` column  

2. **Data Preprocessing**  
   - Sampled legitimate transactions to balance with fraud transactions  
   - Merged both datasets using `pd.concat()`  
   - Shuffled the new dataset for fairness  

3. **Model Training**  
   - Split data into training and testing sets (80/20 split)  
   - Trained a **Logistic Regression** model  

4. **Model Evaluation**  
   - Evaluated results using:
     - **Accuracy Score**
     - **Confusion Matrix**
     - **Precision**, **Recall**, and **F1-Score**

---

## 📊 Results

| Metric | Score |
|--------|-------:|
| Accuracy | 98.7% |
| Precision (Fraud) | 94% |
| Recall (Fraud) | 91% |

✅ The model successfully detects most fraudulent transactions while keeping false alarms low.

---

## 📂 Dataset Link : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

