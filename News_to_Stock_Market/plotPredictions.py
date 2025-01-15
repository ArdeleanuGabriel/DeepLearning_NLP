import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader

from News_to_Stock_Market.training import StockNewsDataset, StockPricePredictor, prepare_data


# Assuming you have the same StockNewsDataset, create_sliding_window, prepare_data functions

def plot_predictions_with_stock_prices(stock_data, predictions, output_file):
    # Adding the predictions to stock_data DataFrame
    stock_data["predictions"] = predictions

    # Create a color array for plotting red (loss) or green (gain) points
    stock_data["color"] = stock_data["predictions"].apply(lambda x: 'green' if x == 1 else 'red')

    # Plot the stock price and red/green points
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['date'], stock_data['close'], label='Stock Price', color='blue', alpha=0.7)
    plt.scatter(stock_data['date'], stock_data['close'], c=stock_data['color'], label='Predictions', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Predictions vs Actual')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Prediction plot saved to {output_file}")


def predict_and_plot(stock_file_path, model_file, sequence_length, feature_columns, output_file, news_data_path):
    # Load stock data
    stock_data = pd.read_csv(stock_file_path)
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.sort_values('date')

    # Load news data
    news_data = pd.read_parquet(news_data_path)
    news_data['date'] = news_data['time'].dt.date

    # Prepare the data by merging stock and news data, and creating features/targets
    X, _ = prepare_data(stock_data, news_data, sequence_length)

    # Set up device and model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(feature_columns)
    hidden_dim = 764
    output_dim = 2
    num_layers = 10

    # Initialize the model
    model = StockPricePredictor(input_dim, hidden_dim, output_dim, num_layers).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Create dataset and dataloader
    dataset = StockNewsDataset(X, np.zeros(X.shape[0]))  # dummy y as we only predict
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
            print(predictions)

    # Prepare stock data for plotting (after sliding window)
    stock_data = stock_data.iloc[sequence_length:].reset_index(drop=True)
    stock_data['predictions'] = predictions

    # Plot the predictions with stock prices
    plot_predictions_with_stock_prices(stock_data, predictions, output_file)


if __name__ == "__main__":
    stock_folder = "SelectedHistoricalFolder"
    news_file = "df_0_news_with_keywords_and_sentiment.parquet"

    trained_model_path = "AAPL.csv_model_weights"
    sequence_length = 5  # same as used during training

    feature_columns = ["text_sentiment"] + [f"kw_{i}" for i in range(100)]  # replace 100 with actual number of keywords

    # for file in os.listdir(stock_folder):
    #     if file.endswith(".csv"):
    stock_file_path = os.path.join("SelectedHistoricalFolder", "AAPL.csv")

    # Prediction and plotting
    predict_and_plot(stock_file_path, trained_model_path, sequence_length, feature_columns, "AAPL_prediction_plot.png", news_file)
