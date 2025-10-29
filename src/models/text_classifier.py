# Nội dung cho tệp: src/models/text_classifier.py

from typing import List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator

Vectorizer = Union[TfidfVectorizer, CountVectorizer]

class TextClassifier:


    def __init__(self, vectorizer: BaseEstimator):
        self.vectorizer = vectorizer
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        print("Mô hình đã được huấn luyện thành công.")

    def predict(self, texts: List[str]) -> List[int]:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics