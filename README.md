Text Classification using TF-IDF & Logistic Regression

This project implements a multi-class text classification model using TF-IDF vectorization and Logistic Regression in Python.
The pipeline includes text cleaning, feature extraction, model training, evaluation, and saving the trained model for reuse.

Project Structure
PyCharmMiscProject/
│
├── train_model.py        # Main training script
├── dataset.csv           # Input dataset (text + labels)
├── model.pkl             # Saved trained Logistic Regression model
├── vectorizer.pkl        # Saved TF-IDF vectorizer
├── README.md             # Project documentation
└── .venv/                # Virtual environment

Dataset Details

Shape: (11314, 3)

Columns:

text → raw input text

label → class label (0–19)

(optional metadata column)

The task is to classify text into 20 different classes.

Workflow (Step-by-Step)

Load Dataset

Reads the CSV file using pandas

Confirms dataset shape

Text Cleaning

Lowercasing

Removing punctuation and numbers

Removing stopwords using NLTK

Feature Extraction

TF-IDF Vectorizer

Max features: 5000

Output shape: (11314, 5000)

Train-Test Split

Standard train/test split using sklearn

Model Training

Algorithm: Logistic Regression

Multi-class classification

Model successfully trained

Evaluation

Precision, Recall, F1-score

Overall Accuracy: 68%

Model Saving

Trained model saved as model.pkl

TF-IDF vectorizer saved as vectorizer.pkl

Model Performance Summary

Accuracy: 68%

Macro Avg F1-Score: 0.68

Weighted Avg F1-Score: 0.69

The model performs reasonably well across most classes, with some classes needing improvement due to overlap or imbalance.

How to Run the Project
1. Create & Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate

2. Install Dependencies
pip install pandas numpy scikit-learn nltk joblib

3. Run Training Script
python train_model.py

Future Improvements
Use LinearSVC or XGBoost
Apply class balancing
Add lemmatization
Hyperparameter tuning (GridSearchCV)
Try word embeddings (Word2Vec / BERT)

Built using Python, Scikit-learn, and NLTK

