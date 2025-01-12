import pandas as pd

file_path = "df_0_news_with_keywords_and_sentiment.parquet"

if __name__ == '__main__':
    df = pd.read_parquet(file_path, columns=["time"])

    df["time"] = pd.to_datetime(df["time"])

    start_date = df["time"].min().strftime("%B %d, %Y")
    end_date = df["time"].max().strftime("%B %d, %Y")

    print(f"Date range: {start_date} to {end_date}")
