
import os
from datasets import load_dataset
import pandas as pd

def main():

    print("Đang tải bộ dữ liệu 'zeroshot/twitter-financial-news-sentiment'...")
    
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()
    
    full_df = pd.concat([train_df, val_df])
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_path = os.path.join(data_dir, 'twitter_financial_sentiment.csv')
    
    full_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"Đã lưu thành công {len(full_df)} bản ghi vào tệp: {csv_path}")
    print("Hoàn tất chuẩn bị dữ liệu.")

if __name__ == "__main__":
    main()