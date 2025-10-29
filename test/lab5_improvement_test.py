import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("--- Chạy thử nghiệm cải tiến (So sánh Logistic Regression và Naive Bayes) ---")

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

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, zero_division=0)

    print("\n--- So sánh hiệu suất ---")
    print(f"Tập kiểm tra (thực tế): {y_test}")

    print("\nBaseline (Logistic Regression):")
    print(f"  Dự đoán: {y_pred_lr}")
    print(f"  Accuracy: {acc_lr:.4f}")
    print(f"  F1-Score: {f1_lr:.4f}")

    print("\nImprovement (Naive Bayes):")
    print(f"  Dự đoán: {y_pred_nb}")
    print(f"  Accuracy: {acc_nb:.4f}")
    print(f"  F1-Score: {f1_nb:.4f}")
    print("---------------------------------")

if __name__ == "__main__":
    main()
