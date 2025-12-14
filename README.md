# Fake Review Detector Using NLP and Machine Learning

This project is a Fake Review Detection System that classifies online reviews as either CG (Computer-Generated / Fake) or OR (Original / Real) using Natural Language Processing (NLP) and Machine Learning. It includes a Streamlit web application, a preprocessing pipeline, and evaluation of multiple ML algorithms to determine the best model.

---

## Features

- Detects Fake vs. Real reviews  
- NLP preprocessing (tokenization, stopword removal, stemming)  
- Feature engineering (character count, word count, sentence count)  
- Machine learning pipeline (TF-IDF + Logistic Regression)  
- Comparison across 12+ machine learning algorithms  
- Streamlit interface with probability confidence meters  

---

## Project Structure
- app.py - Streamlit application
- fake_reviews_pipe.pkl - Trained ML pipeline
- model_metrics.json - Accuracy and precision metrics
- fake-review-detection.ipynb - Model training notebook
- fake-review-multiple-ml-test.ipynb - ML Models testing notebook


---

## Model Performance (Percentages)

| Model | Accuracy (%) | Precision (%) |
|-------|--------------|----------------|
| KNN | 67.74 | 67.26 |
| Naive Bayes (NB) | 79.81 | 77.44 |
| Decision Tree (DT) | 69.09 | 67.41 |
| Logistic Regression (LR) | 88.56 | 87.95 |
| Random Forest (RF) | 84.96 | 85.10 |
| AdaBoost | 76.70 | 75.75 |
| Bagging Classifier (BgC) | 82.28 | 82.05 |
| Extra Trees (ETC) | 84.65 | 85.17 |
| Gradient Boosting (GBDT) | 80.31 | 78.18 |
| XGBoost (XGB) | 87.15 | 86.89 |
| Multinomial NB | 79.81 | 77.44 |

**Final Model Selected:** Logistic Regression  
Chosen due to the highest accuracy and precision.

---

## How It Works

### 1. Text Preprocessing
- Lowercasing  
- Tokenization  
- Stopword and punctuation removal  
- Stemming using PorterStemmer  
- Reconstructed cleaned text  

### 2. Feature Engineering
- Number of characters  
- Number of words  
- Number of sentences  
- Cleaned (transformed) text  

### 3. Prediction
The ML pipeline outputs:
- Label (Fake / Real)  
- Fake and Real probability scores  

---

## Dataset
This project uses the Fake Reviews Dataset from Kaggle.

**Citation:** Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022).
"Creating and detecting fake reviews of online products." Journal of Retailing and Consumer Services, 64, 102771. https://doi.org/10.1016/j.jretconser.2021.102771

**Dataset Link:** https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

---

## Technologies Used
- Python  
- NLTK  
- Scikit-Learn
- Streamlit  
- Pandas / NumPy
- Pickle serialization

---

## Created by:
Shakif Niaz
