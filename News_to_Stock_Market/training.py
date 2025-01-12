import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class StockNewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_window(data, sequence_length, feature_columns, target_col):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        window = data[i:i + sequence_length][feature_columns].values
        X.append(window)
        y.append(data.iloc[i + sequence_length][target_col])
    return np.array(X), np.array(y)


def prepare_data(stock_data, news_data, sequence_length=5):
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    news_data["date"] = pd.to_datetime(news_data["date"])

    data = pd.merge(stock_data, news_data, on="date", how="left")

    data.fillna({"text_sentiment": 0, "key_words": "", "title": ""}, inplace=True)

    scaler = StandardScaler()
    data["text_sentiment"] = scaler.fit_transform(data[["text_sentiment"]])

    data["price_change"] = (data["close"] > data["open"]).astype(int)

    feature_columns = ["text_sentiment"]

    X, y = create_sliding_window(data, sequence_length, feature_columns, "price_change")
    return X, y


class StockPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return self.sigmoid(out)


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            predictions = (outputs >= 0.5).float()
            correct += torch.sum((predictions == y_batch).int()).item()
            total += y_batch.size(0)
            epoch_loss += loss.item()

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            predictions = (outputs >= 0.5).float()
            correct += torch.sum((predictions == y_batch).int()).item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    stock_folder = "SelectedHistoricalFolder"
    news_file = "df_0_news_with_keywords_and_sentiment.parquet"

    all_stock_data = []
    for file in os.listdir(stock_folder):
        if file.endswith(".csv"):
            stock_file_path = os.path.join(stock_folder, file)
            stock_data = pd.read_csv(stock_file_path)
            stock_data["date"] = pd.to_datetime(stock_data["date"])
            all_stock_data.append(stock_data)

    stock_data = pd.concat(all_stock_data, ignore_index=True)

    news_data = pd.read_parquet(news_file)
    news_data["date"] = news_data["time"].dt.date

    sequence_length = 5
    X, y = prepare_data(stock_data, news_data, sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = StockNewsDataset(X_train, y_train)
    test_dataset = StockNewsDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[2]
    hidden_dim = 256
    output_dim = 1
    num_layers = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockPricePredictor(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    evaluate_model(model, test_loader, device)
