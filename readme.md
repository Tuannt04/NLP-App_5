# =========================================================
# BÁO CÁO THỰC NGHIỆM LAB 5 — TEXT CLASSIFICATION
# =========================================================

# =========================================================
# 1. Implementation Steps (Các bước triển khai)
# =========================================================
# - Dữ liệu được tải và làm sạch bằng prepare_dataset.py.
# - Mô hình TextClassifier (TF-IDF + LogisticRegression) được cài trong text_classifier.py.
# - Các thử nghiệm gồm:
#   * lab5_test.py: mô hình cơ sở (baseline).
#   * lab5_improvement_test.py: mô hình cải tiến với Naive Bayes.
#   * lab5_spark_sentiment_analysis.py: pipeline Spark với TF-IDF và Word2Vec.
# - Đánh giá bằng Accuracy và F1-score.

# =========================================================
# 2. Code Execution Guide (Hướng dẫn chạy mã)
# =========================================================
# Thực hiện tuần tự trong main.ipynb:
# 
# !python "dataset/prepare_dataset.py"            # Chuẩn bị dữ liệu
# !python "test/lab5_test.py"                     # Baseline
# !python "test/lab5_improvement_test.py"         # Cải tiến
# !python "test/lab5_spark_sentiment_analysis.py" # Spark pipelines
#
# Kết quả (Accuracy, F1-score) được in trực tiếp trong output notebook.

# =========================================================
# 3. Result Analysis (Phân tích kết quả)
# =========================================================

# ---------------------------------------------------------
# 3.1. Hiệu suất mô hình cơ sở (Baseline Model)
# ---------------------------------------------------------
# - Mô hình: Logistic Regression + TF-IDF
# - Accuracy: ~0.85
# - F1-score: ~0.84
# - Nhận xét:
#   * TF-IDF biểu diễn văn bản dựa trên tần suất từ, hiệu quả cho dữ liệu ngắn.
#   * Logistic Regression cho kết quả ổn định và dễ huấn luyện.
#   * Hạn chế: không hiểu ngữ nghĩa, dễ sai với câu phủ định hoặc mơ hồ.
#   * Ví dụ lỗi: “Không tệ” bị phân loại sai thành tiêu cực.

# ---------------------------------------------------------
# 3.2. Hiệu suất mô hình cải tiến (Improved Model)
# ---------------------------------------------------------
# - Mô hình 1: TF-IDF + Naive Bayes
#   * Accuracy: ~0.87
#   * F1-score: ~0.86
#   * Ưu điểm: nhanh hơn, phù hợp dữ liệu có đặc trưng rời rạc.
# 
# - Mô hình 2: Word2Vec + Logistic Regression
#   * Accuracy: ~0.88
#   * F1-score: ~0.87
#   * Ưu điểm: hiểu được quan hệ ngữ nghĩa giữa các từ, giảm lỗi phủ định.
#   * Hạn chế: thời gian huấn luyện dài hơn, yêu cầu tài nguyên cao.

# ---------------------------------------------------------
# 3.3. So sánh kết quả
# ---------------------------------------------------------
# | Mô hình | Accuracy | F1-score | Nhận xét |
# |----------|-----------|-----------|-----------|
# | Logistic Regression (Baseline) | 0.85 | 0.84 | Ổn định, đơn giản, huấn luyện nhanh. |
# | Naive Bayes (Improved 1) | 0.87 | 0.86 | Cải thiện nhẹ nhờ giả định độc lập giữa đặc trưng. |
# | Word2Vec + Logistic Regression (Improved 2) | 0.88 | 0.87 | Hiểu ngữ nghĩa, kết quả tốt nhất. |

# - Các mô hình cải tiến đều vượt baseline 2–3%.
# - Word2Vec cho hiệu quả cao nhất vì học được quan hệ ngữ nghĩa giữa các từ.
# - Naive Bayes cải thiện ít hơn nhưng huấn luyện nhanh, phù hợp môi trường giới hạn tài nguyên.

# ---------------------------------------------------------
# 3.4. Phân tích nguyên nhân cải thiện
# ---------------------------------------------------------
# 1. TF-IDF chỉ dựa trên tần suất, bỏ qua ngữ cảnh.
# 2. Naive Bayes tận dụng xác suất điều kiện giữa từ và nhãn → tăng tính phân biệt.
# 3. Word2Vec biểu diễn từ trong không gian liên tục → mô hình hiểu được ngữ nghĩa.
# 4. Do đó, cải tiến giúp mô hình:
#    - Tổng quát hóa tốt hơn cho từ chưa gặp.
#    - Giảm lỗi với câu phủ định hoặc đồng nghĩa.
#    - Tuy nhiên, Word2Vec cần nhiều tài nguyên hơn TF-IDF.

# =========================================================
# 4. Challenges and Solutions (Thách thức và cách giải quyết)
# =========================================================
# | Thách thức | Giải pháp |
# |-------------|-----------|
# | Dữ liệu nhiễu, không cân bằng | Làm sạch văn bản, dùng trọng số lớp. |
# | Hiệu suất Spark chậm | Giảm vector size, cache hợp lý. |
# | Tối ưu tham số mô hình | Dùng CrossValidator hoặc thử thủ công với regParam. |

# =========================================================
# 5. References (Tài liệu tham khảo)
# =========================================================
# 1. Apache Spark MLlib Documentation — https://spark.apache.org/docs/latest/ml-guide.html
# 2. Scikit-learn Documentation — https://scikit-learn.org/stable/
# 3. Hugging Face Datasets — https://huggingface.co/docs/datasets
# 4. Mikolov et al. (2013) — Word2Vec: Efficient Estimation of Word Representations in Vector Space.
# 5. Tài liệu học phần “Machine Learning with Spark – Lab 5: Text Classification”.
