<img width="1144" height="132" alt="image" src="https://github.com/user-attachments/assets/dcb70383-563c-48f8-badc-25ec38446c79" /># Comment Category Prediction

## 📌 Project Overview

Online discussion platforms receive a large volume of user-generated comments every day. To maintain quality, safety, and proper moderation, these platforms automatically classify comments into different internal categories based on their content and context.

This project focuses on building a **machine learning system that mimics this real-world moderation pipeline** by predicting the final category assigned to each comment.

The dataset contains not only the **raw text of comments**, but also additional signals such as:

* User engagement (upvotes and downvotes)
* Presence of emoticons
* System-generated indicators related to sensitive topics (race, religion, gender, disability)
* Hidden internal features used by the platform

The main challenge is to **combine unstructured text data with structured numerical and categorical features** to accurately predict the comment’s category.

To solve this, a complete ML pipeline was developed that includes:

* Text preprocessing and TF-IDF feature extraction
* Feature engineering from metadata
* Handling class imbalance
* Training multiple models (linear and tree-based)
* Building an ensemble model for improved performance

The final solution achieves strong predictive performance and reflects how real-world systems integrate **NLP + structured data** for decision-making.

This project demonstrates practical skills in:

* End-to-end machine learning workflow
* Natural Language Processing (NLP)
* Feature engineering
* Model selection and evaluation

---


---

## 🎯 Objective

To build a robust machine learning model that predicts the **final category (`label`)** assigned to each comment using:

* Comment text
* User engagement signals (upvotes/downvotes)
* Emoticon indicators
* Internal platform features
* Sensitive topic indicators (race, religion, etc.)

---

## 📊 Dataset Description
The dataset is not included due to size limitations.

You can download it from:
👉 https://www.kaggle.com/competitions/comment-category-prediction-challenge


### Files:

* `train.csv` — labeled dataset
* `test.csv` — unlabeled dataset
* `sample_submission.csv` — submission format

### Target Variable:

* `label` → 4 classes (0, 1, 2, 3)

---

## 🔍 Exploratory Data Analysis

### Class Distribution

* Label 0 → ~57.7% (majority class)
* Label 2 → ~31.5%
* Label 1 → ~8.0%
* Label 3 → ~2.8% (highly underrepresented)

👉 **Key Observation:** Severe class imbalance — especially Label 3.

---

### Feature Insights

#### Correlation Analysis

* `if_2` → strongest numerical signal (~0.23 correlation)
* `upvote`, `downvote` → very weak predictors
* Emoticon features → negligible linear impact

👉 Conclusion: **Text features are expected to dominate performance**

---

## ⚙️ Feature Engineering

### Time-Based Features

From `created_date`:

* `hour`
* `dayofweek`

---

### Text Processing

* Lowercasing
* URL removal
* TF-IDF vectorization (8000+ features)

👉 Core signal source for classification

---

## 🔧 Preprocessing Pipeline

Implemented using **Pipeline + ColumnTransformer**

### Numerical Features

* Median Imputation
* Standard Scaling

### Categorical Features

* Most Frequent Imputation
* One-Hot Encoding

### Key Advantage

* Prevents **data leakage**
* Ensures consistent transformation for train & test

---

## Train-Test Split

* 80% Training / 20% Validation
* `stratify=y` used to preserve class distribution

👉 Critical due to **class imbalance**

---

##  Model Training & Performance

### 🔹 Model 1 — SGD Classifier (Baseline)

* **Accuracy:** 88.77%
* Works well with sparse TF-IDF features
* Fast and efficient baseline

---

### 🔹 Model 2 — Logistic Regression (Tuned)

* **Accuracy:** 89.63%
* Best Params: `C=10.0`, `penalty='l2'`

👉 Improvement over SGD due to better regularization

---

### 🔹 Model 3 — LightGBM (Champion Model)

* **Accuracy:** 91.05%
* Captures **non-linear relationships**
* Significant improvement over linear models

---

##  Final Model — Ensemble

### Soft Voting Ensemble:

* Tuned LightGBM
* XGBoost

### Why Ensemble?

* Improves generalization
* Helps handle minority class (Label 3)
* Reduces individual model bias

---

## 📈 Kaggle Performance

* ✅ **Public Score:** **0.802**

---

##  Key Learnings

* TF-IDF features are highly effective for text classification
* Class imbalance significantly affects model behavior
* Linear models perform well but plateau quickly
* Tree-based models capture deeper feature interactions
* Ensemble methods improve performance, especially for rare classes

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* LightGBM, XGBoost
* TF-IDF Vectorization
* Matplotlib, Seaborn

---

## 📁 Project Structure

```
├── train.csv
├── test.csv
├── sample_submission.csv
├── notebook.ipynb
├── README.md
```

---

##  How to Run

```bash
git clone https://github.com/Kanishthika11/comment-category-prediction
cd comment-category-prediction
pip install -r requirements.txt
jupyter notebook
```

---

## 📌 Final Insights

* Text data is the **primary driver of prediction accuracy**
* Engineered features (time + engagement) add incremental gains
* LightGBM outperformed linear models due to non-linear capability
* Ensemble learning improved robustness across all classes

---

##  Acknowledgement

This project was completed as part of my **Machine Learning Practices Course** in the BS Degree in Data Science program.

---

## 🔗 GitHub

👉 https://github.com/Kanishthika11

---

