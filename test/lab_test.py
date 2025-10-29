# Nội dung cho tệp: test/lab5_test.py

import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.text_classifier import TextClassifier

def main():

    print("--- Chạy Thử nghiệm Cơ bản (Baseline Logistic Regression) ---")

    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0] 
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.33, random_state=42
    )
    print(f"Kích thước tập huấn luyện: {len(X_train)}")
    print(f"Kích thước tập kiểm tra: {len(X_test)}")

    vectorizer = TfidfVectorizer()
    classifier = TextClassifier(vectorizer=vectorizer)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    metrics = classifier.evaluate(y_test, y_pred)
    
    print("\n--- Kết quả Đánh giá (Baseline) ---")
    print(f"Dự đoán: {y_pred}")
    print(f"Thực tế:  {y_test}")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    main()