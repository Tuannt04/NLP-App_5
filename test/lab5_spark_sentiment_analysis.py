import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def create_spark():
    return (
        SparkSession.builder.appName("AdvancedSentimentAnalysis")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_and_prepare_data(spark: SparkSession, path: str):
    print(f"Đang tải dữ liệu từ {path}...")
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True,
        multiLine=True,
        quote='"',
        escape='"'
    ).dropna(subset=["text", "label"])

    cleaned = (
        df.withColumn("clean_text", lower(col("text")))
        .withColumn("clean_text", regexp_replace(col("clean_text"), r"http\S+", ""))
        .withColumn("clean_text", regexp_replace(col("clean_text"), r"@[a-zA-Z0-9_]+", ""))
        .withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-zA-Z\s]", ""))
    )

    return cleaned.select(col("clean_text").alias("text"), col("label").cast("double"))


def build_evaluators():
    return (
        MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label"),
        MulticlassClassificationEvaluator(metricName="f1", labelCol="label"),
    )


def build_pipelines():
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    hashing = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    lr = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")
    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")

    tokenizer_w2v = Tokenizer(inputCol="text", outputCol="words")
    remover_w2v = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    w2v = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="features")
    lr_w2v = LogisticRegression(maxIter=10, regParam=0.01, featuresCol="features", labelCol="label")

    return {
        "Baseline (TF-IDF + LR)": Pipeline(stages=[tokenizer, remover, hashing, idf, lr]),
        "Improvement (TF-IDF + NB)": Pipeline(stages=[tokenizer, remover, hashing, idf, nb]),
        "Improvement (Word2Vec + LR)": Pipeline(stages=[tokenizer_w2v, remover_w2v, w2v, lr_w2v]),
    }


def main():
    spark = create_spark()
    data_path = "data/twitter_financial_sentiment.csv"

    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy tệp {data_path}.")
        print("Vui lòng chạy 'prepare_dataset.py' trước.")
        spark.stop()
        return

    df = load_and_prepare_data(spark, data_path)
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    print("Đã chia dữ liệu thành train và test.")

    acc_eval, f1_eval = build_evaluators()
    pipelines = build_pipelines()
    results = {}

    for name, pipe in pipelines.items():
        print(f"\n--- Huấn luyện: {name} ---")
        model = pipe.fit(train_data)
        print(f"--- Đánh giá: {name} ---")
        preds = model.transform(test_data)
        results[name] = {
            "Accuracy": acc_eval.evaluate(preds),
            "F1-Score": f1_eval.evaluate(preds),
        }

    print("\n--- KẾT QUẢ SO SÁNH HIỆU SUẤT ---")
    print("-" * 50)
    print(f"{'Mô hình':<30} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:<10.4f} | {metrics['F1-Score']:<10.4f}")
    print("-" * 50)

    spark.stop()


if __name__ == "__main__":
    main()
